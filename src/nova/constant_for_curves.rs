use ark_bn254::g1::Config as BNConfig;
use ark_bn254::g1::G1Affine as g1;
use ark_bn254::g2::G2Affine as g2;
use ark_bn254::{Bn254, Fq, Fr, G2Projective};
use ark_ec::short_weierstrass::Projective;
use crate::nova::pederson::PedersenCommitment;

/// Since we use the cycle of curves (Bn254, Grumpkin) throughout our tests, we define some types here, so later we can easily use them in out tests

pub type E = Bn254;

pub type ScalarField = Fr;

pub type BaseField = Fq;

pub type G1Affine = g1;

pub type G2Affine = g2;

pub type G1 = BNConfig;

pub type G1Projective = Projective<G1>;

pub type C1 = PedersenCommitment<G1Projective>;

pub type C2 = PedersenCommitment<G2Projective>;