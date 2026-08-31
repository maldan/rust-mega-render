//! Parses and validates every visualizer WGSL shader with `naga`, so a typo
//! or type error (e.g. a bad struct layout change) fails `cargo test`
//! instead of only surfacing at runtime when a real GPU device loads it.

fn validate(label: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|e| panic!("{label}: wgsl parse failed: {e}"));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("{label}: wgsl validation failed: {e}"));
}

macro_rules! wgsl_test {
    ($name:ident, $path:literal) => {
        #[test]
        fn $name() {
            validate($path, include_str!(concat!("../src/visualizer/wgpu/", $path)));
        }
    };
}

wgsl_test!(tess_wgsl_is_valid, "tess.wgsl");
wgsl_test!(mesh_wgsl_is_valid, "mesh.wgsl");
wgsl_test!(hair_wgsl_is_valid, "hair.wgsl");
wgsl_test!(shadow_wgsl_is_valid, "shadow.wgsl");
wgsl_test!(debug_wgsl_is_valid, "debug.wgsl");
wgsl_test!(skybox_wgsl_is_valid, "skybox.wgsl");
wgsl_test!(ssao_wgsl_is_valid, "ssao.wgsl");
wgsl_test!(gtao_wgsl_is_valid, "gtao.wgsl");
wgsl_test!(contact_shadow_wgsl_is_valid, "contact_shadow.wgsl");
wgsl_test!(ssgi_wgsl_is_valid, "ssgi.wgsl");
wgsl_test!(ssr_wgsl_is_valid, "ssr.wgsl");
wgsl_test!(ssr_temporal_wgsl_is_valid, "ssr_temporal.wgsl");
wgsl_test!(hiz_copy_wgsl_is_valid, "hiz_copy.wgsl");
wgsl_test!(hiz_downsample_wgsl_is_valid, "hiz_downsample.wgsl");
wgsl_test!(ssgi_temporal_wgsl_is_valid, "ssgi_temporal.wgsl");
wgsl_test!(ssgi_upsample_wgsl_is_valid, "ssgi_upsample.wgsl");
wgsl_test!(ssgi_atrous_wgsl_is_valid, "ssgi_atrous.wgsl");
wgsl_test!(ssgi_bounce_wgsl_is_valid, "ssgi_bounce.wgsl");
wgsl_test!(copy_wgsl_is_valid, "copy.wgsl");
wgsl_test!(blur_wgsl_is_valid, "blur.wgsl");
wgsl_test!(bloom_wgsl_is_valid, "bloom.wgsl");
wgsl_test!(composite_wgsl_is_valid, "composite.wgsl");
wgsl_test!(fxaa_wgsl_is_valid, "fxaa.wgsl");
wgsl_test!(dof_wgsl_is_valid, "dof.wgsl");
wgsl_test!(dof_prelit_wgsl_is_valid, "dof_prelit.wgsl");
wgsl_test!(dof_up_wgsl_is_valid, "dof_up.wgsl");
wgsl_test!(dof_temporal_wgsl_is_valid, "dof_temporal.wgsl");
wgsl_test!(mb_dilate_wgsl_is_valid, "mb_dilate.wgsl");
wgsl_test!(mb_gather_wgsl_is_valid, "mb_gather.wgsl");
wgsl_test!(hud_wgsl_is_valid, "hud.wgsl");
wgsl_test!(debug_blit_wgsl_is_valid, "debug_blit.wgsl");
