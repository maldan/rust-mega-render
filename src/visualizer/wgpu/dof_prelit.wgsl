//! Bake AO / contact / SSGI / SSR into HDR color before DOF.

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0),
    );
    let xy = p[vi];
    var o: VsOut;
    o.pos = vec4(xy, 0.0, 1.0);
    o.uv = vec2(xy.x * 0.5 + 0.5, 0.5 - xy.y * 0.5);
    return o;
}

struct PrelitUniforms {
    /// ao, contact, ssgi, ssr intensities
    intensities: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: PrelitUniforms;
@group(0) @binding(1) var scene_tex: texture_2d<f32>;
@group(0) @binding(2) var ao_tex: texture_2d<f32>;
@group(0) @binding(3) var contact_tex: texture_2d<f32>;
@group(0) @binding(4) var ssgi_tex: texture_2d<f32>;
@group(0) @binding(5) var ssr_tex: texture_2d<f32>;
@group(0) @binding(6) var samp: sampler;
@group(0) @binding(7) var albedo_tex: texture_2d<f32>;
@group(0) @binding(8) var orm_tex: texture_2d<f32>;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4<f32> {
    var color = textureSampleLevel(scene_tex, samp, i.uv, 0.0).rgb;
    let inten = u.intensities;

    if inten.x > 0.0 {
        let ao = textureSampleLevel(ao_tex, samp, i.uv, 0.0).r;
        color *= mix(1.0, ao, clamp(inten.x, 0.0, 2.0));
    }
    if inten.y > 0.0 {
        let cs = textureSampleLevel(contact_tex, samp, i.uv, 0.0).r;
        color *= mix(1.0, cs, clamp(inten.y, 0.0, 2.0));
    }
    if inten.z > 0.0 {
        let irr = textureSampleLevel(ssgi_tex, samp, i.uv, 0.0).rgb;
        let albedo = textureSampleLevel(albedo_tex, samp, i.uv, 0.0).rgb;
        let metallic = textureSampleLevel(orm_tex, samp, i.uv, 0.0).b;
        var contrib = irr * albedo * (1.0 - metallic) * inten.z;
        let y = luma(contrib);
        contrib = contrib / (1.0 + y * 0.35);
        color += contrib;
    }
    if inten.w > 0.0 {
        color += textureSampleLevel(ssr_tex, samp, i.uv, 0.0).rgb * inten.w;
    }
    return vec4(color, 1.0);
}
