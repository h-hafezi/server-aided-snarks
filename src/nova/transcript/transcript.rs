use crate::nova::gadgets::non_native::util::{convert_affine_to_scalars, cast_field};
use ark_crypto_primitives::sponge::Absorb;
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use crate::nova::poseidon::PoseidonHash;

#[derive(Clone)]
pub struct Transcript<F: PrimeField + Absorb> {
    // This will hold the current state of the transcript
    pub state: F,
    // the poseidon hash
    pub poseidon_hash: PoseidonHash<F>,
}

impl<F: PrimeField + Absorb> Transcript<F> {
    pub fn new(_label: &'static [u8]) -> Transcript<F> {
        Transcript {
            state: F::ONE,
            poseidon_hash: PoseidonHash::new(),
        }
    }
}

/// the current implementation discards all labels
impl<F: PrimeField + Absorb> Transcript<F> {
    pub fn append_u64(&mut self, _label: &'static [u8], n: u64) {
        let f = F::from(n);
        self.append_scalar(_label, &f);
    }

    pub fn append_message(&mut self, _label: &'static [u8], _msg: &[u8]) {
        // I'm not sure if it's important to implement this
    }

    pub fn append_scalar(&mut self, _label: &'static [u8], scalar: &F) {
        self.poseidon_hash.update_sponge(vec![scalar.clone()]);
    }

    pub fn append_scalar_non_native<Q: PrimeField>(&mut self, _label: &'static [u8], scalar: &Q) {
        // convert into F
        let converted_scalar = cast_field::<Q, F>(scalar.clone());
        self.poseidon_hash.update_sponge(vec![converted_scalar.clone()]);
    }


    pub fn append_scalars(&mut self, _label: &'static [u8], scalars: &[F]) {
        for f in scalars {
            self.append_scalar(_label, &f);
        }
    }

    pub fn append_scalars_non_native<Q: PrimeField>(&mut self, _label: &'static [u8], scalars: &[Q]) {
        for q in scalars {
            self.append_scalar_non_native(_label, q);
        }
    }


    pub fn challenge_scalar(&mut self, _label: &'static [u8]) -> F {
        let new_state = self.poseidon_hash.output();
        self.state = new_state;
        self.append_scalar(_label, &new_state);
        new_state
    }

    pub fn challenge_vector(&mut self, _label: &'static [u8], len: usize) -> Vec<F> {
        let mut res = Vec::with_capacity(len);
        for _ in 0..len {
            res.push(self.challenge_scalar(_label));
        }
        res
    }

    /// appends E::G1Affine points, by first converting them through convert_affine_to_scalars
    pub fn append_point<E: Pairing<ScalarField=F>>(&mut self, label: &'static [u8], point: &E::G1Affine)
    where
        <<E as Pairing>::G1Affine as ark_ec::AffineRepr>::BaseField: PrimeField,
    {
        let (x, y) = convert_affine_to_scalars::<E>(*point);
        self.append_scalar(label, &x);
        self.append_scalar(label, &y);
    }

    /// calls append_point multiple times
    pub fn append_points<E: Pairing<ScalarField=F>>(&mut self, label: &'static [u8], points: &[E::G1Affine])
    where
        <<E as Pairing>::G1Affine as ark_ec::AffineRepr>::BaseField: PrimeField,
    {
        for p in points {
            self.append_point::<E>(label, p);
        }
    }
}

pub trait AppendToTranscript<F: PrimeField + Absorb> {
    fn append_to_transcript(&self, label: &'static [u8], transcript: &mut Transcript<F>);
}