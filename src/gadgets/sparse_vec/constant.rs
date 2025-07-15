#[derive(Debug)]
pub enum Error {
    NOutOfRange,
}

/// for different N we get the sparsity of n
pub fn get_t(n: usize) -> Result<usize, Error> {
    // Define the mapping of n = 2^i to t
    // Using a vector indexed by i-10 since i ranges from 10 to 20
    let t_values = [
        62, // 2^10
        60, // 2^11
        58, // 2^12
        55, // 2^13
        53, // 2^14
        50, // 2^15
        47, // 2^16
        45, // 2^17
        42, // 2^18
        40, // 2^19
        40, // 2^20
    ];

    // Check bounds on n, must be in [2^10, 2^20]
    let min_n = 1 << 10; // 2^10
    let max_n = 1 << 20; // 2^20
    if n < min_n || n > max_n {
        return Err(Error::NOutOfRange);
    }

    // Find i such that 2^i <= n < 2^{i+1}
    // i ranges from 10 to 20
    let i = (usize::BITS as usize - n.leading_zeros() as usize - 1).max(10);

    // Determine the index into t_values
    // If n is exactly 2^i, return t for 2^i
    // Else if 2^i < n < 2^{i+1}, return t for 2^{i+1}
    // Handle edge cases if i >= 20
    let index = if n == (1 << i) {
        i - 10
    } else {
        // If n > 2^i, pick t for 2^{i+1}
        if i + 1 > 20 {
            // n > 2^20 but we already bounded n <= 2^20, so this is error
            return Err(Error::NOutOfRange);
        }
        i + 1 - 10
    };

    let t = t_values[index];

    Ok(t)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_powers_of_two() {
        let test_cases = [
            (1 << 10, 62),
            (1 << 11, 60),
            (1 << 12, 58),
            (1 << 13, 55),
            (1 << 14, 53),
            (1 << 15, 50),
            (1 << 16, 47),
            (1 << 17, 45),
            (1 << 18, 42),
            (1 << 19, 40),
            (1 << 20, 40),
        ];

        for (n, expected_t) in test_cases {
            let result = get_t(n);
            assert!(result.is_ok(), "Expected Ok for n={}, got {:?}", n, result);
            assert_eq!(result.unwrap(), expected_t, "Incorrect t for n={}", n);
        }
    }

    #[test]
    fn test_between_powers_of_two() {
        let test_cases = [
            ((1 << 10) + 1, 60),      // between 2^10 and 2^11
            ((1 << 11) + 5, 58),      // between 2^11 and 2^12
            ((1 << 12) + 100, 55),    // between 2^12 and 2^13
            ((1 << 15) + 5000, 47),   // between 2^15 and 2^16
            ((1 << 19) + 100000, 40),// between 2^19 and 2^20
        ];

        for (n, expected_t) in test_cases {
            let result = get_t(n);
            assert!(result.is_ok(), "Expected Ok for n={}, got {:?}", n, result);
            assert_eq!(result.unwrap(), expected_t, "Incorrect t for n={}", n);
        }
    }

    #[test]
    fn test_out_of_range() {
        let test_cases = [
            (1 << 9),        // less than 2^10
            ((1 << 20) + 1), // greater than 2^20
            0,
            usize::MAX,
        ];

        for n in test_cases {
            let result = get_t(n);
            assert!(result.is_err(), "Expected Err for n={}, got {:?}", n, result);
            assert!(matches!(result, Err(Error::NOutOfRange)), "Expected NOutOfRange error for n={}", n);
        }
    }
}
