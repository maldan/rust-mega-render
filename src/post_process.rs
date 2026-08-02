#[derive(Clone, Debug)]
pub struct PostProcessSettings {
    pub ssao: SsaoSettings,
    pub bloom: BloomSettings,
}

#[derive(Clone, Debug)]
pub struct SsaoSettings {
    pub enabled: bool,
    pub radius: f32,
    pub bias: f32,
    pub intensity: f32,
}

#[derive(Clone, Debug)]
pub struct BloomSettings {
    pub enabled: bool,
    pub threshold: f32,
    pub intensity: f32,
}

impl Default for PostProcessSettings {
    fn default() -> Self {
        Self {
            ssao: SsaoSettings {
                enabled: false,
                radius: 1.0,
                bias: 0.05,
                intensity: 1.0,
            },
            bloom: BloomSettings {
                enabled: false,
                threshold: 0.7,
                intensity: 0.6,
            },
        }
    }
}

impl PostProcessSettings {
    pub fn any_enabled(&self) -> bool {
        self.ssao.enabled || self.bloom.enabled
    }
}
