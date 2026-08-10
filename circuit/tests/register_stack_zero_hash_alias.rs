use anyhow::Result;
use circuit::hash_utils::CircuitBuilderHashUtils;
use circuit::types::config::{Builder, C, F};
use circuit::types::constants::{REGISTER_STACK_SIZE, TRANSFER_ASSET};
use circuit::types::register::{
    BaseRegisterInfo, RegisterInfoTargetWitness, RegisterStack, RegisterStackTarget,
};
use plonky2::field::types::Field64;
use plonky2::iop::witness::PartialWitness;
use plonky2::plonk::circuit_data::CircuitConfig;

#[test]
fn nonempty_transfer_instruction_has_empty_stack_hash_before_and_after_pop() -> Result<()> {
    let mut builder = Builder::new(CircuitConfig::standard_recursion_config());
    let mut stack = RegisterStackTarget::new(&mut builder);
    let witness_stack = stack;

    let empty_hash = builder.zero_hash_out();
    let old_hash = stack.hash(&mut builder);
    builder.connect_hashes(old_hash, empty_hash);

    let enabled = builder._true();
    builder.conditional_assert_eq_constant(
        enabled,
        stack[0].instruction_type,
        TRANSFER_ASSET as u64,
    );
    stack.pop_front(&mut builder, enabled);

    let new_hash = stack.hash(&mut builder);
    builder.connect_hashes(new_hash, empty_hash);

    let data = builder.build::<C>();

    // These are all ordinary, in-range transfer instruction fields. Two fields
    // unused by INTERNAL_TRANSFER make the is_empty sum exactly Goldilocks' order.
    let mut forged = BaseRegisterInfo::empty();
    forged.instruction_type = TRANSFER_ASSET;
    forged.account_index = 1001;
    forged.pending_size = 50;
    forged.pending_type = 1;
    // is_empty checks generic_field_0 separately, so this forged transfer is
    // specifically a transfer to account zero (the protocol treasury).
    forged.generic_field_0 = 0;
    forged.generic_field_2 = 3;
    let visible_sum = i128::from(forged.instruction_type)
        + i128::from(forged.account_index)
        + i128::from(forged.pending_size)
        + i128::from(forged.pending_type)
        + i128::from(forged.generic_field_2);
    forged.pending_expiry = i64::MAX;
    forged.generic_field_3 =
        i64::try_from(i128::from(F::ORDER) - visible_sum - i128::from(forged.pending_expiry))?;
    assert!(!forged.is_empty());

    let mut forged_stack = RegisterStack {
        stack: [BaseRegisterInfo::empty(); REGISTER_STACK_SIZE],
        count: 1,
    };
    forged_stack.stack[0] = forged;

    let mut pw = PartialWitness::<F>::new();
    pw.set_register_info_target(&witness_stack, &forged_stack)?;

    let proof = data.prove(pw)?;
    data.verify(proof)?;
    Ok(())
}
