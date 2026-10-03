"""Selected source transformations. Original PSMS snapshot remains unchanged."""
from cascade import sky_source, replace_once

def replace_function(src, signature, body):
    start = src.index(signature)
    left = src.index('{', start)
    depth = 1
    right = left + 1
    while depth:
        depth += (src[right] == '{') - (src[right] == '}')
        right += 1
    return src[:left] + '{\n' + body + '\n}' + src[right:]


def lookup_sources(src):
    # Spherical atmosphere: diffuse source depends only on radius and solar zenith.
    header = src[:src.index('void mainImage')]
    lut = header + '''
void main() {
    float h = (gl_FragCoord.x - 0.5) / 511.0 * 0.1;
    float mu = clamp((gl_FragCoord.y - 0.5) / 255.0 * 2.0 - 1.0,-0.999999,0.999999);
    float r = planetRadius + h;
    vec3 L = vec3(sqrt(max(0.0,1.0-mu*mu)),0.0,mu);
    outColor = vec4(PSMSGather(vec3(0.0,0.0,r),vec3(0.0,0.0,1.0),L),1.0);
}
'''
    # A 32x32 cell table rebuilt for this camera/time; no per-ray register array.
    event = header + '''
void main() {
    ivec2 pix = ivec2(gl_FragCoord.xy);
    vec2 cell = floor(uCamera.xy / HSE_CELL_SIZE) - 16.0 + vec2(pix.x/2,pix.y);
    float cloudTime = iTime*100.0+5000.0;
    vec2 r0 = HSEHash22(cell+7.13), r1 = HSEHash22(cell+23.91), r2 = HSEHash22(cell+51.37);
    float u = fract(cloudTime*HSE_TIME_SCALE+HSEHash12(cell+37.11));
    float life = HSELife(u);
    vec2 center = (cell+mix(vec2(0.25),vec2(0.75),r0))*HSE_CELL_SIZE;
    center += (r1*2.0-1.0)*(650.0*(u*2.0-1.0));
    float radius = mix(HSE_MIN_RADIUS,HSE_MAX_RADIUS,r1.y)*HSERadiusLife(u);
    float aspect = mix(0.68,1.0,r2.y);
    vec2 radii = r0.x < 0.5 ? vec2(radius,radius*aspect) : vec2(radius*aspect,radius);
    float amp = (HSEHash12(cell+19.71) < 0.38 || life <= 0.0001) ? 0.0 : mix(700.0,HSE_MAX_RISE,r2.x)*life;
    outColor = pix.x % 2 == 0 ? vec4(center,max(radii,vec2(1.0))) : vec4(amp,0.0,0.0,1.0);
}
'''
    # Keep exact evaluation for table-domain excursions.
    src = src.replace('float HSEEventRise(', 'float HSEEventRiseExact(', 1)
    at = src.index('//================ NEXT: per-cell event cache')
    src = src[:at] + '''
float HSEEventRise(vec2 p, float cloudTime, vec2 cell) {
    ivec2 q = ivec2(cell - (floor(uCamera.xy/HSE_CELL_SIZE)-16.0));
    if (any(lessThan(q,ivec2(0))) || any(greaterThanEqual(q,ivec2(32))))
        return HSEEventRiseExact(p,cloudTime,cell);
    float amp = texelFetch(uEvents,ivec2(q.x*2+1,q.y),0).x;
    if (amp == 0.0) return 0.0;
    vec4 e = texelFetch(uEvents,ivec2(q.x*2,q.y),0);
    vec2 v = (p-e.xy)/e.zw;
    float rSq = dot(v,v);
    if (rSq >= 1.0) return 0.0;
    float s = 1.0-rSq;
    return amp*(s*s*s);
}
''' + src[at:]
    src = replace_function(src, 'float3 PSMSGather(', '''
    vec2 f = vec2(clamp((length(P)-planetRadius)/0.1,0.0,1.0),clamp(dot(up,L)*0.5+0.5,0.0,1.0));
    return texture(uGather,(f*vec2(511.0,255.0)+0.5)/vec2(512.0,256.0)).rgb;
''')
    src = src.replace('uniform sampler2D uProbes;', 'uniform sampler2D uProbes, uGather, uEvents;')
    return src, lut, event


def shared_sources():
    src = sky_source()
    start = src.index('    float3 probeCam = Game2Atm_Alt(CameraPosition);', src.index('void mainImage'))
    end = src.index('\n    vec3 c = Sky(', start)
    probes = src[start:end]
    header = src[:src.index('void mainImage')]
    probe = header + '''
void main() {
    vec3 CameraPosition = uCamera;
    vSunLightDir = uSun;
''' + probes + '''
    outColor = vec4(gl_FragCoord.x < 1.0 ? gAmbZenith : gAmbHorizon, 1.0);
}
'''
    src = src[:start] + '''
    float3 probeCam = Game2Atm_Alt(CameraPosition);
    gAmbZenith = texelFetch(uProbes, ivec2(0,0), 0).rgb;
    gAmbHorizon = texelFetch(uProbes, ivec2(1,0), 0).rgb;
''' + src[end:]
    src = src.replace('uniform vec2 uTargetSize;', 'uniform vec2 uTargetSize;\nuniform sampler2D uProbes;')
    return src, probe


def candidate_sources(src):
    header = src[:src.index('void mainImage')]
    prepass = header + '''
void main() {
    vec2 origin = (floor(uCamera.xy/HSE_CELL_SIZE)-16.0)*HSE_CELL_SIZE;
    vec2 lo = origin + floor(gl_FragCoord.xy)*500.0;
    vec2 hi = lo+500.0;
    vec2 cell = floor((lo+250.0)/HSE_CELL_SIZE);
    float upper[9];
    float bestLower=0.0;
    for(int k=0;k<9;k++) {
        ivec2 q=ivec2(cell+vec2(k%3-1,k/3-1)-(floor(uCamera.xy/HSE_CELL_SIZE)-16.0));
        upper[k]=0.0;
        if(any(lessThan(q,ivec2(0)))||any(greaterThanEqual(q,ivec2(32)))) {
            // Outside the event table: mark this tile for exact fallback.
            outColor=vec4(-1.0); return;
        }
        float amp=texelFetch(uEvents,ivec2(q.x*2+1,q.y),0).x;
        vec4 e=texelFetch(uEvents,ivec2(q.x*2,q.y),0);
        vec2 nearD=(clamp(e.xy,lo,hi)-e.xy)/e.zw;
        vec2 farD=max(abs(lo-e.xy),abs(hi-e.xy))/e.zw;
        float u=max(1.0-dot(nearD,nearD),0.0);
        float l=max(1.0-dot(farD,farD),0.0);
        upper[k]=amp*u*u*u;
        bestLower=max(bestLower,amp*l*l*l);
    }
    int mask=0;
    for(int k=0;k<9;k++)
        if(upper[k]>0.0 && upper[k]+0.01>=bestLower) mask|=1<<k;
    outColor=vec4(float(mask),0.0,0.0,1.0);
}
'''
    src = src.replace('float HSELowCloudTop(', 'float HSELowCloudTopExact(',1)
    at = src.index('// Apply an already-evaluated local roof')
    src = src[:at] + '''
float HSELowCloudTop(vec2 p, float cloudTime, float baseTop) {
    vec2 origin=(floor(uCamera.xy/HSE_CELL_SIZE)-16.0)*HSE_CELL_SIZE;
    ivec2 q=ivec2(floor((p-origin)/500.0));
    if(any(lessThan(q,ivec2(0)))||any(greaterThanEqual(q,ivec2(512))))
        return HSELowCloudTopExact(p,cloudTime,baseTop);
    int mask=int(texelFetch(uCandidates,q,0).x);
    if(mask<0) return HSELowCloudTopExact(p,cloudTime,baseTop);
    float rise=0.0;
    vec2 baseCell=floor(p/HSE_CELL_SIZE);
    while(mask!=0) {
        int k=findLSB(mask);
        mask &= mask-1;
        rise=max(rise,HSEEventRise(p,cloudTime,baseCell+vec2(k%3-1,k/3-1)));
    }
    float broad=(noise2d(vec3(p*0.000055,cloudTime*0.00002))-0.5)*90.0;
    return baseTop+max(0.0,rise+broad);
}
''' + src[at:]
    src = src.replace('uniform sampler2D uProbes, uGather, uEvents;',
                      'uniform sampler2D uProbes, uGather, uEvents, uCandidates;')
    return src, prepass


def bounded_source(src):
    # Cloud profile 0 fades at 40 km. View anchors extend < 240 m in XY;
    # shadow anchors < 4 km. This is inside the table's > 120 km margin.
    src=replace_once(src,'if (d.x >= 0.0) {','if (d.x >= 0.0 && d.x < a.fade) {')
    src=replace_once(src,'''    if (any(lessThan(q,ivec2(0))) || any(greaterThanEqual(q,ivec2(32))))
        return HSEEventRiseExact(p,cloudTime,cell);''','')
    src=replace_once(src,'''    if(any(lessThan(q,ivec2(0)))||any(greaterThanEqual(q,ivec2(512))))
        return HSELowCloudTopExact(p,cloudTime,baseTop);''','')
    src=replace_once(src,'''    if(mask<0) return HSELowCloudTopExact(p,cloudTime,baseTop);''','')
    return src
