use crate::nova::commitment::{Commitment, CommitmentScheme};
use crate::nova::gadgets::non_native::util::cast_field;
use crate::nova::gadgets::r1cs::{R1CSInstance, RelaxedR1CSInstance};
use ark_ec::short_weierstrass::{Projective, SWCurveConfig};
use ark_ec::{AffineRepr};
use ark_ff::PrimeField;

// Functions below are used by Nova, since in Nova we require to hash R1CS instances
impl<G: SWCurveConfig, C: CommitmentScheme<Projective<G>>> R1CSInstance<G, C>
where
    G::BaseField: PrimeField
{
    /// This function should be consistent with the counterpart for R1CSInstanceVar as below
    pub fn to_sponge_field_elements(&self) -> Vec<G::ScalarField> {
        let mut res = Vec::<G::ScalarField>::new();

        // append vector X
        res.extend(self.X.as_slice());

        // convert group into native field elements
        let w = self.commitment_W.into_affine().xy().unwrap();
        let x = cast_field::<G::BaseField, G::ScalarField>(w.0);
        let y = cast_field::<G::BaseField, G::ScalarField>(w.1);
        res.extend(vec![x, y]);

        res
    }
}

impl<G: SWCurveConfig, C: CommitmentScheme<Projective<G>>> RelaxedR1CSInstance<G, C>
where
    G::BaseField: PrimeField
{
    /// This function should be consistent with the counterpart for RelaxedR1CSInstanceVar as below
    pub fn to_sponge_field_elements(&self) -> Vec<G::ScalarField> {
        let mut res = Vec::<G::ScalarField>::new();

        // append vector X
        res.extend(self.X.as_slice());

        // convert group into native field elements
        let w = self.commitment_W.into_affine().xy().unwrap();
        let x = cast_field::<G::BaseField, G::ScalarField>(w.0);
        let y = cast_field::<G::BaseField, G::ScalarField>(w.1);
        res.extend(vec![x, y]);

        // convert group into native field elements
        let e = self.commitment_E.into_affine().xy().unwrap();
        let x = cast_field::<G::BaseField, G::ScalarField>(e.0);
        let y = cast_field::<G::BaseField, G::ScalarField>(e.1);
        res.extend(vec![x, y]);

        res
    }
}
