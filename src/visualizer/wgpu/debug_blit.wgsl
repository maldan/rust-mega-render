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

struct BlitUniforms {
    // x = mode, y = exposure, z = near, w = far
    params: vec4<f32>,
    // x = ao intensity (for Debug Ao view), yzw unused
    ao_params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: BlitUniforms;
@group(0) @binding(1) var color_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var orm_tex: texture_2d<f32>;
@group(0) @binding(4) var depth_tex: texture_depth_2d;
@group(0) @binding(5) var ao_tex: texture_2d<f32>;
@group(0) @binding(6) var samp: sampler;

fn tonemap_reinhard(c: vec3<f32>) -> vec3<f32> {
    return c / (c + vec3(1.0));
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4<f32> {
    let mode = u32(u.params.x);
    let exposure = u.params.y;
    let near = u.params.z;
    let far = u.params.w;

    if mode == 0u || mode == 1u {
        // Final preview / scene color: expose + Reinhard (full post is separate).
        var c = textureSampleLevel(color_tex, samp, i.uv, 0.0).rgb * exposure;
        c = tonemap_reinhard(c);
        return vec4(c, 1.0);
    }
    if mode == 2u {
        let n = textureSampleLevel(normal_tex, samp, i.uv, 0.0).xyz;
        return vec4(n * 0.5 + 0.5, 1.0);
    }
    if mode == 3u {
        let r = textureSampleLevel(orm_tex, samp, i.uv, 0.0).g;
        return vec4(vec3(r), 1.0);
    }
    if mode == 4u {
        let m = textureSampleLevel(orm_tex, samp, i.uv, 0.0).b;
        return vec4(vec3(m), 1.0);
    }
    if mode == 5u {
        let d = textureSample(depth_tex, samp, i.uv);
        // Rough linearization for DirectX-style clip depth.
        let z = near * far / max(far - d * (far - near), 1e-5);
        let t = clamp(z / far, 0.0, 1.0);
        return vec4(vec3(t), 1.0);
    }
    if mode == 8u {
        // SSGI HDR preview — expose + Reinhard.
        let intensity = max(u.ao_params.x, 0.0);
        var c = textureSampleLevel(ao_tex, samp, i.uv, 0.0).rgb * exposure * intensity;
        c = tonemap_reinhard(c);
        return vec4(c, 1.0);
    }
    if mode == 9u {
        // SSR HDR specular preview — expose + Reinhard.
        let intensity = max(u.ao_params.x, 0.0);
        var c = textureSampleLevel(ao_tex, samp, i.uv, 0.0).rgb * exposure * intensity;
        c = tonemap_reinhard(c);
        return vec4(c, 1.0);
    }
    if mode == 10u {
        // Albedo G-buffer (passed via ao_tex binding in Albedo debug mode).
        let a = textureSampleLevel(ao_tex, samp, i.uv, 0.0).rgb;
        return vec4(a, 1.0);
    }
    if mode == 11u {
        // DOF CoC: magenta = near, green = far (already encoded).
        let c = textureSampleLevel(ao_tex, samp, i.uv, 0.0).rgb;
        return vec4(c, 1.0);
    }
    if mode == 12u {
        // DOF HDR result — expose + Reinhard.
        var c = textureSampleLevel(ao_tex, samp, i.uv, 0.0).rgb * exposure;
        c = tonemap_reinhard(c);
        return vec4(c, 1.0);
    }
    // AO (6) / Contact shadow (7) — intensity matches Final composite.
    let ao = textureSampleLevel(ao_tex, samp, i.uv, 0.0).r;
    let intensity = clamp(u.ao_params.x, 0.0, 2.0);
    let shown = mix(1.0, ao, intensity);
    return vec4(vec3(shown), 1.0);
}
