//! Black-76 price, greeks and implied vol (guide §3.4–3.5). Port of `core/quant/black76.py`.

pub const IV_LO: f64 = 0.01;
pub const IV_HI: f64 = 5.0;
const IV_TOL: f64 = 1e-10;
const IV_MAXIT: usize = 200;

#[allow(non_snake_case)]
pub fn N(x: f64) -> f64 {
    0.5 * libm::erfc(-x / std::f64::consts::SQRT_2)
}

pub fn n(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Price and greeks. With `t <= 0` or `sigma <= 0` only `price` (discounted intrinsic) is set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Greeks {
    pub price: f64,
    pub delta: f64,
    pub delta_spot: f64,
    pub gamma_1pct: f64,
    pub vega_1pt: f64,
    pub theta_1d: f64,
    pub p_itm: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Black76 {
    Intrinsic { price: f64 },
    Full(Greeks),
}

impl Black76 {
    pub fn price(&self) -> f64 {
        match self {
            Self::Intrinsic { price } => *price,
            Self::Full(g) => g.price,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct NonPositiveInput;

#[allow(clippy::many_single_char_names, clippy::too_many_arguments)]
pub fn black76(
    f: f64,
    k: f64,
    t: f64,
    sigma: f64,
    r: f64,
    is_call: bool,
    s: f64,
) -> Result<Black76, NonPositiveInput> {
    if !(f > 0.0 && k > 0.0 && s > 0.0) {
        return Err(NonPositiveInput);
    }
    let df = (-r * t).exp();
    if t <= 0.0 || sigma <= 0.0 {
        let intr = if is_call {
            (f - k).max(0.0)
        } else {
            (k - f).max(0.0)
        };
        return Ok(Black76::Intrinsic { price: df * intr });
    }
    let sd = sigma * t.sqrt();
    let d1 = ((f / k).ln() + 0.5 * sd * sd) / sd;
    let d2 = d1 - sd;
    let (price, delta, p_itm) = if is_call {
        (df * (f * N(d1) - k * N(d2)), df * N(d1), N(d2))
    } else {
        (df * (k * N(-d2) - f * N(-d1)), -df * N(-d1), N(-d2))
    };
    let fs = f / s;
    let gamma_spot = df * n(d1) / (f * sd) * fs * fs;
    Ok(Black76::Full(Greeks {
        price,
        delta,
        delta_spot: delta * fs,
        gamma_1pct: gamma_spot * s * 0.01,
        vega_1pt: df * f * n(d1) * t.sqrt() / 100.0,
        theta_1d: (-df * f * n(d1) * sigma / (2.0 * t.sqrt()) + r * price) / 365.0,
        p_itm,
    }))
}

/// Bisection on [0.01, 5.0]. `Err(reason)` instead of a clamped bound (guide §13).
#[allow(clippy::too_many_arguments)]
pub fn implied_vol(
    f: f64,
    k: f64,
    t: f64,
    price: f64,
    r: f64,
    is_call: bool,
    s: f64,
) -> Result<f64, &'static str> {
    let df = (-r * t).exp();
    let intrinsic = df
        * if is_call {
            (f - k).max(0.0)
        } else {
            (k - f).max(0.0)
        };
    let upper = df * if is_call { f } else { k };
    if price <= intrinsic {
        return Err("below_intrinsic");
    }
    if price >= upper {
        return Err("above_max_price");
    }
    let diff = |sigma: f64| -> Result<f64, &'static str> {
        black76(f, k, t, sigma, r, is_call, s)
            .map(|b| b.price() - price)
            .map_err(|_| "non_positive_input")
    };
    let (mut a, mut b) = (IV_LO, IV_HI);
    if diff(a)? > 0.0 {
        return Err("below_min_vol");
    }
    if diff(b)? < 0.0 {
        return Err("above_max_vol");
    }
    for _ in 0..IV_MAXIT {
        let m = 0.5 * (a + b);
        let fm = diff(m)?;
        if fm.abs() < IV_TOL || (b - a) < 1e-12 {
            return Ok(m);
        }
        if fm > 0.0 {
            b = m;
        } else {
            a = m;
        }
    }
    Ok(0.5 * (a + b))
}
