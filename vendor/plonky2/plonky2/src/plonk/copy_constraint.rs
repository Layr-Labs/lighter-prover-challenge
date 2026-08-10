use crate::iop::target::Target;

/// A copy constraint.
#[derive(Debug)]
pub struct CopyConstraint {
    pub pair: (Target, Target),
}

impl From<(Target, Target)> for CopyConstraint {
    fn from(pair: (Target, Target)) -> Self {
        Self { pair }
    }
}
