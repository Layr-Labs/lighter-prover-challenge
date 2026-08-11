.arch armv9-a+sme2
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_gl_fft_fused2_ssve             ; -- Begin function gl_fft_fused2_ssve
	.p2align	2
_gl_fft_fused2_ssve:                    ; @gl_fft_fused2_ssve
	.cfi_startproc
; %bb.0:
	stp	d15, d14, [sp, #-96]!           ; 16-byte Folded Spill
	.cfi_def_cfa_offset 96
	stp	d13, d12, [sp, #16]             ; 16-byte Folded Spill
	stp	d11, d10, [sp, #32]             ; 16-byte Folded Spill
	stp	d9, d8, [sp, #48]               ; 16-byte Folded Spill
	stp	x22, x21, [sp, #64]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #80]             ; 16-byte Folded Spill
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset b8, -40
	.cfi_offset b9, -48
	.cfi_offset b10, -56
	.cfi_offset b11, -64
	.cfi_offset b12, -72
	.cfi_offset b13, -80
	.cfi_offset b14, -88
	.cfi_offset b15, -96
	smstart	sm
	ptrue	p0.d
	rdvl	x10, #1
	mov	w8, #4                          ; =0x4
	lsl	x8, x8, x2
	cmp	x8, x1
	b.hi	LBB0_6
; %bb.1:
	mov	w9, #1                          ; =0x1
	lsl	x9, x9, x2
	lsr	x10, x10, #4
	lsl	x10, x10, #2
	cmp	x10, x9
	b.hi	LBB0_6
; %bb.2:
	add	x11, x4, x9, lsl #3
	lsl	x14, x9, #3
	add	x12, x0, x14
	lsl	x13, x8, #3
	addvl	x17, x14, #1
	add	x14, x0, x17
	mov	w15, #24                        ; =0x18
	madd	x15, x9, x15, x0
	add	x16, x0, x9, lsl #4
	add	x17, x4, x17
	ptrue	p1.b
	ptrue	p2.d
	mov	z0.d, #0xffffffff
	mov	z1.s, #-1                       ; =0xffffffffffffffff
	mov	z2.d, #0                        ; =0x0
	mov	x2, x8
LBB0_3:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_4 Depth 2
	mov	x5, #0                          ; =0x0
	mov	x6, x10
LBB0_4:                                 ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	add	x7, x3, x5
	ld1b	{ z17.b }, p1/z, [x3, x5]
	ld1d	{ z18.d }, p2/z, [x7, #1, mul vl]
	ld1b	{ z5.b }, p1/z, [x12, x5]
	ld1b	{ z6.b }, p1/z, [x14, x5]
	add	x7, x15, x5
	ld1b	{ z19.b }, p1/z, [x15, x5]
	ld1d	{ z20.d }, p2/z, [x7, #1, mul vl]
	add	x19, x16, x5
	ld1b	{ z16.b }, p1/z, [x16, x5]
	ld1d	{ z7.d }, p2/z, [x19, #1, mul vl]
	add	x20, x0, x5
	ld1b	{ z4.b }, p1/z, [x0, x5]
	ld1d	{ z3.d }, p2/z, [x20, #1, mul vl]
	mul	z21.d, z17.d, z5.d
	umulh	z5.d, z17.d, z5.d
	lsr	z22.d, z5.d, #32
	sub	z23.d, z21.d, z22.d
	cmphi	p3.d, p0/z, z22.d, z21.d
	sub	z23.d, p3/m, z23.d, z0.d
	umullb	z21.d, z5.s, z1.s
	add	z5.d, z23.d, z21.d
	cmphi	p3.d, p0/z, z21.d, z5.d
	add	z5.d, p3/m, z5.d, z0.d
	mul	z21.d, z18.d, z6.d
	umulh	z6.d, z18.d, z6.d
	lsr	z22.d, z6.d, #32
	sub	z23.d, z21.d, z22.d
	cmphi	p3.d, p0/z, z22.d, z21.d
	sub	z23.d, p3/m, z23.d, z0.d
	umullb	z21.d, z6.s, z1.s
	add	z6.d, z23.d, z21.d
	cmphi	p3.d, p0/z, z21.d, z6.d
	add	z6.d, p3/m, z6.d, z0.d
	mul	z21.d, z17.d, z19.d
	umulh	z17.d, z17.d, z19.d
	lsr	z19.d, z17.d, #32
	sub	z22.d, z21.d, z19.d
	cmphi	p3.d, p0/z, z19.d, z21.d
	sub	z22.d, p3/m, z22.d, z0.d
	umullb	z17.d, z17.s, z1.s
	add	z19.d, z22.d, z17.d
	cmphi	p3.d, p0/z, z17.d, z19.d
	add	z19.d, p3/m, z19.d, z0.d
	mul	z17.d, z18.d, z20.d
	umulh	z18.d, z18.d, z20.d
	lsr	z20.d, z18.d, #32
	sub	z21.d, z17.d, z20.d
	cmphi	p3.d, p0/z, z20.d, z17.d
	sub	z21.d, p3/m, z21.d, z0.d
	umullb	z17.d, z18.s, z1.s
	add	z18.d, z21.d, z17.d
	cmphi	p3.d, p0/z, z17.d, z18.d
	add	z18.d, p3/m, z18.d, z0.d
	add	z17.d, z16.d, z19.d
	cmphi	p3.d, p0/z, z16.d, z17.d
	mov	z20.d, z17.d
	add	z20.d, p3/m, z20.d, z0.d
	cmphi	p3.d, p0/z, z17.d, z20.d
	add	z20.d, p3/m, z20.d, z0.d
	add	z17.d, z7.d, z18.d
	cmphi	p3.d, p0/z, z7.d, z17.d
	mov	z21.d, z17.d
	add	z21.d, p3/m, z21.d, z0.d
	cmphi	p3.d, p0/z, z17.d, z21.d
	add	z21.d, p3/m, z21.d, z0.d
	sub	z17.d, z16.d, z19.d
	cmphi	p3.d, p0/z, z19.d, z16.d
	sel	z16.d, p3, z0.d, z2.d
	sub	z19.d, z17.d, z16.d
	cmphi	p3.d, p0/z, z16.d, z17.d
	sub	z19.d, p3/m, z19.d, z0.d
	sub	z16.d, z7.d, z18.d
	cmphi	p3.d, p0/z, z18.d, z7.d
	sel	z7.d, p3, z0.d, z2.d
	sub	z22.d, z16.d, z7.d
	cmphi	p3.d, p0/z, z7.d, z16.d
	sub	z22.d, p3/m, z22.d, z0.d
	add	x21, x4, x5
	ld1b	{ z7.b }, p1/z, [x4, x5]
	ld1d	{ z16.d }, p2/z, [x21, #1, mul vl]
	ld1b	{ z23.b }, p1/z, [x11, x5]
	ld1b	{ z24.b }, p1/z, [x17, x5]
	mul	z17.d, z7.d, z20.d
	umulh	z7.d, z7.d, z20.d
	lsr	z18.d, z7.d, #32
	sub	z20.d, z17.d, z18.d
	cmphi	p3.d, p0/z, z18.d, z17.d
	sub	z20.d, p3/m, z20.d, z0.d
	umullb	z7.d, z7.s, z1.s
	add	z17.d, z20.d, z7.d
	cmphi	p3.d, p0/z, z7.d, z17.d
	add	z17.d, p3/m, z17.d, z0.d
	mul	z7.d, z16.d, z21.d
	umulh	z16.d, z16.d, z21.d
	lsr	z18.d, z16.d, #32
	sub	z20.d, z7.d, z18.d
	cmphi	p3.d, p0/z, z18.d, z7.d
	sub	z20.d, p3/m, z20.d, z0.d
	umullb	z7.d, z16.s, z1.s
	add	z18.d, z20.d, z7.d
	cmphi	p3.d, p0/z, z7.d, z18.d
	add	z18.d, p3/m, z18.d, z0.d
	mul	z7.d, z23.d, z19.d
	umulh	z16.d, z23.d, z19.d
	lsr	z19.d, z16.d, #32
	sub	z20.d, z7.d, z19.d
	cmphi	p3.d, p0/z, z19.d, z7.d
	sub	z20.d, p3/m, z20.d, z0.d
	umullb	z16.d, z16.s, z1.s
	add	z7.d, z20.d, z16.d
	cmphi	p3.d, p0/z, z16.d, z7.d
	add	z7.d, p3/m, z7.d, z0.d
	mul	z16.d, z24.d, z22.d
	umulh	z19.d, z24.d, z22.d
	lsr	z20.d, z19.d, #32
	sub	z21.d, z16.d, z20.d
	cmphi	p3.d, p0/z, z20.d, z16.d
	sub	z21.d, p3/m, z21.d, z0.d
	umullb	z19.d, z19.s, z1.s
	add	z16.d, z21.d, z19.d
	cmphi	p3.d, p0/z, z19.d, z16.d
	add	z16.d, p3/m, z16.d, z0.d
	add	z19.d, z4.d, z5.d
	cmphi	p3.d, p0/z, z4.d, z19.d
	mov	z20.d, z19.d
	add	z20.d, p3/m, z20.d, z0.d
	cmphi	p3.d, p0/z, z19.d, z20.d
	add	z20.d, p3/m, z20.d, z0.d
	add	z19.d, z3.d, z6.d
	cmphi	p3.d, p0/z, z3.d, z19.d
	mov	z21.d, z19.d
	add	z21.d, p3/m, z21.d, z0.d
	cmphi	p3.d, p0/z, z19.d, z21.d
	add	z21.d, p3/m, z21.d, z0.d
	sub	z19.d, z4.d, z5.d
	cmphi	p3.d, p0/z, z5.d, z4.d
	sel	z4.d, p3, z0.d, z2.d
	sub	z5.d, z19.d, z4.d
	cmphi	p3.d, p0/z, z4.d, z19.d
	sub	z5.d, p3/m, z5.d, z0.d
	sub	z4.d, z3.d, z6.d
	cmphi	p3.d, p0/z, z6.d, z3.d
	sel	z3.d, p3, z0.d, z2.d
	sub	z6.d, z4.d, z3.d
	cmphi	p3.d, p0/z, z3.d, z4.d
	sub	z6.d, p3/m, z6.d, z0.d
	add	z3.d, z20.d, z17.d
	cmphi	p3.d, p0/z, z20.d, z3.d
	mov	z4.d, z3.d
	add	z4.d, p3/m, z4.d, z0.d
	cmphi	p3.d, p0/z, z3.d, z4.d
	add	z4.d, p3/m, z4.d, z0.d
	st1b	{ z4.b }, p1, [x0, x5]
	add	z3.d, z21.d, z18.d
	cmphi	p3.d, p0/z, z21.d, z3.d
	mov	z4.d, z3.d
	add	z4.d, p3/m, z4.d, z0.d
	cmphi	p3.d, p0/z, z3.d, z4.d
	add	z4.d, p3/m, z4.d, z0.d
	st1d	{ z4.d }, p2, [x20, #1, mul vl]
	sub	z3.d, z20.d, z17.d
	cmphi	p3.d, p0/z, z17.d, z20.d
	sel	z4.d, p3, z0.d, z2.d
	sub	z17.d, z3.d, z4.d
	cmphi	p3.d, p0/z, z4.d, z3.d
	sub	z17.d, p3/m, z17.d, z0.d
	st1b	{ z17.b }, p1, [x16, x5]
	sub	z3.d, z21.d, z18.d
	cmphi	p3.d, p0/z, z18.d, z21.d
	sel	z4.d, p3, z0.d, z2.d
	sub	z17.d, z3.d, z4.d
	cmphi	p3.d, p0/z, z4.d, z3.d
	sub	z17.d, p3/m, z17.d, z0.d
	st1d	{ z17.d }, p2, [x19, #1, mul vl]
	add	z3.d, z5.d, z7.d
	cmphi	p3.d, p0/z, z5.d, z3.d
	mov	z4.d, z3.d
	add	z4.d, p3/m, z4.d, z0.d
	cmphi	p3.d, p0/z, z3.d, z4.d
	add	z4.d, p3/m, z4.d, z0.d
	st1b	{ z4.b }, p1, [x12, x5]
	add	z3.d, z6.d, z16.d
	cmphi	p3.d, p0/z, z6.d, z3.d
	mov	z4.d, z3.d
	add	z4.d, p3/m, z4.d, z0.d
	cmphi	p3.d, p0/z, z3.d, z4.d
	add	z4.d, p3/m, z4.d, z0.d
	st1b	{ z4.b }, p1, [x14, x5]
	sub	z3.d, z5.d, z7.d
	cmphi	p3.d, p0/z, z7.d, z5.d
	sel	z4.d, p3, z0.d, z2.d
	sub	z5.d, z3.d, z4.d
	cmphi	p3.d, p0/z, z4.d, z3.d
	sub	z5.d, p3/m, z5.d, z0.d
	st1b	{ z5.b }, p1, [x15, x5]
	sub	z3.d, z6.d, z16.d
	cmphi	p3.d, p0/z, z16.d, z6.d
	sel	z4.d, p3, z0.d, z2.d
	sub	z5.d, z3.d, z4.d
	cmphi	p3.d, p0/z, z4.d, z3.d
	sub	z5.d, p3/m, z5.d, z0.d
	st1d	{ z5.d }, p2, [x7, #1, mul vl]
	incw	x6
	addvl	x5, x5, #2
	cmp	x6, x9
	b.ls	LBB0_4
; %bb.5:                                ;   in Loop: Header=BB0_3 Depth=1
	add	x12, x12, x13
	add	x14, x14, x13
	add	x15, x15, x13
	add	x16, x16, x13
	add	x0, x0, x13
	add	x2, x2, x8
	cmp	x2, x1
	b.ls	LBB0_3
LBB0_6:
	smstop	sm
	ldp	x20, x19, [sp, #80]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #64]             ; 16-byte Folded Reload
	ldp	d9, d8, [sp, #48]               ; 16-byte Folded Reload
	ldp	d11, d10, [sp, #32]             ; 16-byte Folded Reload
	ldp	d13, d12, [sp, #16]             ; 16-byte Folded Reload
	ldp	d15, d14, [sp], #96             ; 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore b8
	.cfi_restore b9
	.cfi_restore b10
	.cfi_restore b11
	.cfi_restore b12
	.cfi_restore b13
	.cfi_restore b14
	.cfi_restore b15
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	_gl_fft_single_ssve             ; -- Begin function gl_fft_single_ssve
	.p2align	2
_gl_fft_single_ssve:                    ; @gl_fft_single_ssve
	.cfi_startproc
; %bb.0:
	stp	d15, d14, [sp, #-64]!           ; 16-byte Folded Spill
	.cfi_def_cfa_offset 64
	stp	d13, d12, [sp, #16]             ; 16-byte Folded Spill
	stp	d11, d10, [sp, #32]             ; 16-byte Folded Spill
	stp	d9, d8, [sp, #48]               ; 16-byte Folded Spill
	.cfi_offset b8, -8
	.cfi_offset b9, -16
	.cfi_offset b10, -24
	.cfi_offset b11, -32
	.cfi_offset b12, -40
	.cfi_offset b13, -48
	.cfi_offset b14, -56
	.cfi_offset b15, -64
	smstart	sm
	ptrue	p0.d
	mov	w8, #1                          ; =0x1
	lsl	x8, x8, x2
	mov	w9, #2                          ; =0x2
	lsl	x9, x9, x2
	rdvl	x10, #1
	lsr	x10, x10, #4
	cmp	x9, x1
	lsl	x10, x10, #2
	ccmp	x10, x8, #2, ls
	b.hi	LBB1_5
; %bb.1:
	lsl	x12, x8, #3
	lsl	x11, x9, #3
	addvl	x12, x12, #1
	ptrue	p1.d
	ptrue	p2.b
	mov	z0.d, #0xffffffff
	mov	z1.s, #-1                       ; =0xffffffffffffffff
	mov	z2.d, #0                        ; =0x0
	mov	x13, x9
LBB1_2:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_3 Depth 2
	mov	x14, x0
	mov	x15, x3
	mov	x16, x10
LBB1_3:                                 ;   Parent Loop BB1_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ld1d	{ z3.d }, p1/z, [x15]
	ld1d	{ z4.d }, p1/z, [x15, #1, mul vl]
	ld1d	{ z5.d }, p1/z, [x14, x8, lsl #3]
	ld1b	{ z6.b }, p2/z, [x14, x12]
	ld1d	{ z7.d }, p1/z, [x14]
	ld1d	{ z16.d }, p1/z, [x14, #1, mul vl]
	mul	z17.d, z3.d, z5.d
	umulh	z3.d, z3.d, z5.d
	lsr	z5.d, z3.d, #32
	sub	z18.d, z17.d, z5.d
	cmphi	p3.d, p0/z, z5.d, z17.d
	sub	z18.d, p3/m, z18.d, z0.d
	umullb	z3.d, z3.s, z1.s
	add	z5.d, z18.d, z3.d
	cmphi	p3.d, p0/z, z3.d, z5.d
	add	z5.d, p3/m, z5.d, z0.d
	mul	z3.d, z4.d, z6.d
	umulh	z4.d, z4.d, z6.d
	lsr	z6.d, z4.d, #32
	sub	z17.d, z3.d, z6.d
	cmphi	p3.d, p0/z, z6.d, z3.d
	sub	z17.d, p3/m, z17.d, z0.d
	umullb	z3.d, z4.s, z1.s
	add	z4.d, z17.d, z3.d
	cmphi	p3.d, p0/z, z3.d, z4.d
	add	z4.d, p3/m, z4.d, z0.d
	add	z3.d, z7.d, z5.d
	cmphi	p3.d, p0/z, z7.d, z3.d
	mov	z6.d, z3.d
	add	z6.d, p3/m, z6.d, z0.d
	cmphi	p3.d, p0/z, z3.d, z6.d
	add	z6.d, p3/m, z6.d, z0.d
	st1d	{ z6.d }, p1, [x14]
	add	z3.d, z16.d, z4.d
	cmphi	p3.d, p0/z, z16.d, z3.d
	mov	z6.d, z3.d
	add	z6.d, p3/m, z6.d, z0.d
	cmphi	p3.d, p0/z, z3.d, z6.d
	add	z6.d, p3/m, z6.d, z0.d
	st1d	{ z6.d }, p1, [x14, #1, mul vl]
	sub	z3.d, z7.d, z5.d
	cmphi	p3.d, p0/z, z5.d, z7.d
	sel	z5.d, p3, z0.d, z2.d
	sub	z6.d, z3.d, z5.d
	cmphi	p3.d, p0/z, z5.d, z3.d
	sub	z6.d, p3/m, z6.d, z0.d
	st1d	{ z6.d }, p1, [x14, x8, lsl #3]
	sub	z3.d, z16.d, z4.d
	cmphi	p3.d, p0/z, z4.d, z16.d
	sel	z4.d, p3, z0.d, z2.d
	sub	z5.d, z3.d, z4.d
	cmphi	p3.d, p0/z, z4.d, z3.d
	sub	z5.d, p3/m, z5.d, z0.d
	st1b	{ z5.b }, p2, [x14, x12]
	incw	x16
	addvl	x15, x15, #2
	addvl	x14, x14, #2
	cmp	x16, x8
	b.ls	LBB1_3
; %bb.4:                                ;   in Loop: Header=BB1_2 Depth=1
	add	x0, x0, x11
	add	x13, x13, x9
	cmp	x13, x1
	b.ls	LBB1_2
LBB1_5:
	smstop	sm
	ldp	d9, d8, [sp, #48]               ; 16-byte Folded Reload
	ldp	d11, d10, [sp, #32]             ; 16-byte Folded Reload
	ldp	d13, d12, [sp, #16]             ; 16-byte Folded Reload
	ldp	d15, d14, [sp], #64             ; 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore b8
	.cfi_restore b9
	.cfi_restore b10
	.cfi_restore b11
	.cfi_restore b12
	.cfi_restore b13
	.cfi_restore b14
	.cfi_restore b15
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols
