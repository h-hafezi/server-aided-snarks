use ark_crypto_primitives::sponge::poseidon::{find_poseidon_ark_and_mds, PoseidonSponge};
use ark_crypto_primitives::sponge::{poseidon::PoseidonConfig, Absorb, CryptographicSponge};
use ark_ff::PrimeField;


#[derive(Clone)]
pub struct PoseidonHash<F: Absorb + PrimeField> {
    pub sponge: PoseidonSponge<F>,
}

pub fn get_poseidon_config<F: PrimeField>() -> PoseidonConfig<F> {
    // 120 bit security target as in
    // https://eprint.iacr.org/2019/458.pdf
    // t = rate + 1

    let full_rounds = 8;
    let partial_rounds = 60;
    let alpha = 5;
    let rate = 4;

    let (ark, mds) = find_poseidon_ark_and_mds::<F>(
        F::MODULUS_BIT_SIZE as u64,
        rate,
        full_rounds,
        partial_rounds,
        0,
    );
    PoseidonConfig::new(
        full_rounds as usize,
        partial_rounds as usize,
        alpha,
        mds,
        ark,
        rate,
        1,
    )
}

impl<F: Absorb + PrimeField> PoseidonHash<F> {
    /// This Poseidon configuration generator agrees with Circom's Poseidon(4) in the case of BN254's scalar field
    pub fn new() -> Self {
        let poseidon_params = get_poseidon_config::<F>();
        Self {
            sponge: PoseidonSponge::new(&poseidon_params),
        }
    }

    pub fn update_sponge<A: Absorb>(&mut self, field_vector: Vec<A>) -> () {
        for field_element in field_vector {
            self.sponge.absorb(&field_element);
        }
    }

    pub fn output(&mut self) -> F {
        let squeezed_field_element: Vec<F> = self.sponge.squeeze_field_elements(1);
        squeezed_field_element[0]
    }
}
