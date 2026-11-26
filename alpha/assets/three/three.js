(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const r of document.querySelectorAll('link[rel="modulepreload"]'))n(r);new MutationObserver(r=>{for(const s of r)if(s.type==="childList")for(const a of s.addedNodes)a.tagName==="LINK"&&a.rel==="modulepreload"&&n(a)}).observe(document,{childList:!0,subtree:!0});function t(r){const s={};return r.integrity&&(s.integrity=r.integrity),r.referrerPolicy&&(s.referrerPolicy=r.referrerPolicy),r.crossOrigin==="use-credentials"?s.credentials="include":r.crossOrigin==="anonymous"?s.credentials="omit":s.credentials="same-origin",s}function n(r){if(r.ep)return;r.ep=!0;const s=t(r);fetch(r.href,s)}})();const eu="modulepreload",tu=function(i,e){return new URL(i,e).href},So={},nu=function(e,t,n){let r=Promise.resolve();if(t&&t.length>0){let l=function(h){return Promise.all(h.map(p=>Promise.resolve(p).then(m=>({status:"fulfilled",value:m}),m=>({status:"rejected",reason:m}))))};const a=document.getElementsByTagName("link"),o=document.querySelector("meta[property=csp-nonce]"),u=o?.nonce||o?.getAttribute("nonce");r=l(t.map(h=>{if(h=tu(h,n),h in So)return;So[h]=!0;const p=h.endsWith(".css"),m=p?'[rel="stylesheet"]':"";if(n)for(let y=a.length-1;y>=0;y--){const b=a[y];if(b.href===h&&(!p||b.rel==="stylesheet"))return}else if(document.querySelector(`link[href="${h}"]${m}`))return;const v=document.createElement("link");if(v.rel=p?"stylesheet":eu,p||(v.as="script"),v.crossOrigin="",v.href=h,u&&v.setAttribute("nonce",u),document.head.appendChild(v),p)return new Promise((y,b)=>{v.addEventListener("load",y),v.addEventListener("error",()=>b(new Error(`Unable to preload CSS for ${h}`)))})}))}function s(a){const o=new Event("vite:preloadError",{cancelable:!0});if(o.payload=a,window.dispatchEvent(o),!o.defaultPrevented)throw a}return r.then(a=>{for(const o of a||[])o.status==="rejected"&&s(o.reason);return e().catch(s)})};async function iu(i={}){var e,t=i,n=!!globalThis.window,r=!!globalThis.WorkerGlobalScope,s=globalThis.process?.versions?.node&&globalThis.process?.type!="renderer";if(s){const{createRequire:c}=await nu(()=>import("./__vite-browser-external.js"),[],import.meta.url);var a=c(import.meta.url)}var o=!1;t.setup=function(){if(o)return;o=!0,t.initTBB();function c(L,N,Z=(ae=>ae)){if(N)for(let ae of N)L.push_back(Z(ae));return L}function f(L=0){return typeof L!="object"&&(L={segments:L}),{...t._DefaultCircularQuality(),...L}}t.segmentsFor=function(L,N){return t._SegmentsFor(f(N),L)},t.forDivisions=function(L,N){return t._ForDivisions(L,N)};function S(L,N=(Z=>Z)){const Z=[],ae=L.size();for(let Te=0;Te<ae;Te++)Z.push(N(L.get(Te)));return Z}function T(L,N,Z){const ae={failed:!1,failure:void 0},Te=vo(L(ae),N);let Ze;try{Ze=Z(Te)}finally{xo(Te)}if(ae.failed)throw Ze.delete(),ae.failure;return Ze}function U(L){const N=L.status();if(N!=="NoError")throw L.delete(),new t.ManifoldError(N);return L}function k(L,N=(Z=>Z)){const Z=[],ae=L.size();for(let Te=0;Te<ae;Te++){const Ze=L.get(Te),bt=Ze.size(),rt=[];for(let Tn=0;Tn<bt;Tn++)rt.push(N(Ze.get(Tn)));Ze.delete(),Z.push(rt)}return Z}function H(L){L[0].length<3&&(L=[L]);const N=new t.Vector2_vec2;for(const Z of L){const ae=c(new t.Vector_vec2,Z,Te=>Te instanceof Array?{x:Te[0],y:Te[1]}:Te);N.push_back(ae),ae.delete()}return N}function X(L){return L[0]instanceof Array?{x:L[0][0],y:L[0][1]}:typeof L[0]=="number"?{x:L[0]||0,y:L[1]||0}:L[0]}function J(L){return L[0]instanceof Array?{x:L[0][0],y:L[0][1],z:L[0][2]}:typeof L[0]=="number"?{x:L[0]||0,y:L[1]||0,z:L[2]||0}:L[0]}function he(L){return L=="Round"?1:L=="Miter"?2:L=="Bevel"?3:0}const pe=t.CrossSection;function ge(L,N){if(L instanceof pe)return N(L);const Z=H(L),ae=new pe(Z);Z.delete();const Te=N(ae);return ae.delete(),Te}t.CrossSection.prototype.translate=function(...L){return this._Translate(X(L))},t.CrossSection.prototype.scale=function(L){return typeof L=="number"?this._Scale({x:L,y:L}):this._Scale(X([L]))},t.CrossSection.prototype.mirror=function(L){return this._Mirror(X([L]))},t.CrossSection.prototype.warp=function(L){return T(N=>function(Z){if(!N.failed)try{const ae=Je(Z,"double"),Te=Je(Z+8,"double"),Ze=[ae,Te];L(Ze),ut(Z,Ze[0],"double"),ut(Z+8,Ze[1],"double")}catch(ae){N.failed=!0,N.failure=ae}},"vi",N=>this._Warp(N))},t.CrossSection.prototype.decompose=function(){const L=this._Decompose(),N=S(L);return L.delete(),N},t.CrossSection.prototype.bounds=function(){const L=this._Bounds();return{min:["x","y"].map(N=>L.min[N]),max:["x","y"].map(N=>L.max[N])}},t.CrossSection.prototype.offset=function(L,N="Round",Z=2,ae=0){return this._Offset(L,he(N),Z,f(ae))},t.CrossSection.prototype.simplify=function(L=0){return this._Simplify(L)},t.CrossSection.prototype.extrude=function(L,N=0,Z=0,ae=[1,1],Te=!1){typeof ae=="number"&&(ae=[ae,ae]),ae=X([ae]);const Ze=this._ToPolygons(),bt=t._Extrude(Ze,L,N,Z,ae);if(Ze.delete(),!Te)return bt;const rt=bt.translate([0,0,-L/2]);return bt.delete(),rt},t.CrossSection.prototype.revolve=function(L=0,N=360){const Z=this._ToPolygons(),ae=t._Revolve(Z,f(L),N);return Z.delete(),ae},t.CrossSection.prototype.add=function(L){return ge(L,N=>this._add(N))},t.CrossSection.prototype.subtract=function(L){return ge(L,N=>this._subtract(N))},t.CrossSection.prototype.intersect=function(L){return ge(L,N=>this._intersect(N))},t.CrossSection.prototype.toPolygons=function(){const L=this._ToPolygons(),N=k(L,Z=>[Z.x,Z.y]);return L.delete(),N},t.Manifold.prototype.warp=function(L){const N=T(Z=>function(ae){if(!Z.failed)try{const Te=Je(ae,"double"),Ze=Je(ae+8,"double"),bt=Je(ae+16,"double"),rt=[Te,Ze,bt];L(rt),ut(ae,rt[0],"double"),ut(ae+8,rt[1],"double"),ut(ae+16,rt[2],"double")}catch(Te){Z.failed=!0,Z.failure=Te}},"vi",Z=>this._Warp(Z));return U(N)},t.Manifold.prototype.warpBatch=function(L){const N=T(Z=>function(ae,Te){if(!Z.failed)try{const Ze=t.HEAPF64??B;if(!Ze)throw new Error("WASM heap is not initialized (HEAPF64 unavailable)");const bt=new Float64Array(Ze.buffer,ae,Te*3).slice();L(bt,Te);const rt=t.HEAPF64??B;new Float64Array(rt.buffer,ae,Te*3).set(bt)}catch(Ze){Z.failed=!0,Z.failure=Ze}},"vii",Z=>this._WarpBatch(Z));return U(N)},t.Manifold.prototype.calculateNormals=function(L=52.5){return this._CalculateNormals(L)},t.Manifold.prototype.smoothByNormals=function(){return this._SmoothByNormals()},t.Manifold.prototype.setProperties=function(L,N){const Z=this.numProp();return T(ae=>function(Te,Ze,bt){if(!ae.failed)try{const rt=[];for(let Bt=0;Bt<L;++Bt)rt[Bt]=Je(Te+8*Bt,"double");const Tn=[];for(let Bt=0;Bt<3;++Bt)Tn[Bt]=Je(Ze+8*Bt,"double");const Rr=[];for(let Bt=0;Bt<Z;++Bt)Rr[Bt]=Je(bt+8*Bt,"double");N(rt,Tn,Rr);for(let Bt=0;Bt<L;++Bt)ut(Te+8*Bt,rt[Bt],"double")}catch(rt){ae.failed=!0,ae.failure=rt}},"viii",ae=>this._SetProperties(L,ae))},t.Manifold.prototype.translate=function(...L){return this._Translate(J(L))},t.Manifold.prototype.rotate=function(L,N,Z){return Array.isArray(L)?this._Rotate(...L):this._Rotate(L,N||0,Z||0)},t.Manifold.prototype.scale=function(L){return typeof L=="number"?this._Scale({x:L,y:L,z:L}):this._Scale(J([L]))},t.Manifold.prototype.mirror=function(L){return this._Mirror(J([L]))},t.Manifold.prototype.trimByPlane=function(L,N=0){return this._TrimByPlane(J([L]),N)},t.Manifold.prototype.slice=function(L=0){const N=this._Slice(L),Z=new pe(N);return N.delete(),Z},t.Manifold.prototype.project=function(){const L=this._Project(),N=new pe(L);return L.delete(),N},t.Manifold.prototype.windingNumber=function(L){const N=new t.Vector_vec3;for(const Te of L)N.push_back(J([Te]));const Z=this._WindingNumber(N);N.delete();const ae=S(Z);return Z.delete(),ae},t.Manifold.prototype.rayCast=function(L,N){const Z=this._RayCast(J([L]),J([N])),ae=S(Z,Te=>({originalID:Te.originalID,faceID:Te.faceID,distance:Te.distance,position:["x","y","z"].map(Ze=>Te.position[Ze]),normal:["x","y","z"].map(Ze=>Te.normal[Ze]),crossing:Te.crossing}));return Z.delete(),ae},t.Manifold.prototype.split=function(L){const N=this._Split(L),Z=S(N);return N.delete(),Z},t.Manifold.prototype.splitByPlane=function(L,N=0){const Z=this._SplitByPlane(J([L]),N),ae=S(Z);return Z.delete(),ae},t.Manifold.prototype.decompose=function(){const L=this._Decompose(),N=S(L);return L.delete(),N},t.Manifold.prototype.boundingBox=function(){const L=this._boundingBox();return{min:["x","y","z"].map(N=>L.min[N]),max:["x","y","z"].map(N=>L.max[N])}},t.Manifold.prototype.simplify=function(L=0){return this._Simplify(L)};class Ce{constructor(N,{numProp:Z=3,triVerts:ae=new Uint32Array,vertProperties:Te=new N,mergeFromVert:Ze,mergeToVert:bt,runIndex:rt,runOriginalID:Tn,faceID:Rr,halfedgeTangent:Bt,runTransform:Jc,runFlags:Qc}={}){this.numProp=Z,this.triVerts=ae,this.vertProperties=Te,this.mergeFromVert=Ze,this.mergeToVert=bt,this.runIndex=rt,this.runOriginalID=Tn,this.faceID=Rr,this.halfedgeTangent=Bt,this.runTransform=Jc,this.runFlags=Qc}get numTri(){return this.triVerts.length/3}get numVert(){return this.vertProperties.length/this.numProp}get numRun(){return this.runOriginalID.length}verts(N){return this.triVerts.subarray(3*N,3*(N+1))}position(N){return this.vertProperties.subarray(this.numProp*N,this.numProp*N+3)}extras(N){return this.vertProperties.subarray(this.numProp*N+3,this.numProp*(N+1))}tangent(N){return this.halfedgeTangent.subarray(4*N,4*(N+1))}transform(N){const Z=new Array(16);for(const ae of[0,1,2,3])for(const Te of[0,1,2])Z[4*ae+Te]=this.runTransform[12*N+3*ae+Te];return Z[15]=1,Z}backside(N){return this.runFlags!=null&&N<this.runFlags.length&&(this.runFlags[N]&1)!==0}hasNormals(N){return this.runFlags!=null&&N<this.runFlags.length&&(this.runFlags[N]&2)!==0}}class Ue extends Ce{constructor(N){super(Float32Array,N)}merge(N=0){return new Ue(t._Merge(this,N))}}class ze extends Ce{constructor(N){super(Float64Array,N)}merge(N=0){return new ze(t._Merge64(this,N))}}const Xe=Symbol.for("manifold-3d.Mesh64");ze.prototype[Xe]=!0,t.Mesh=Ue,t.Mesh64=ze,t.Manifold.prototype.getMesh=function(){return new Ue(this._GetMeshJS())},t.Manifold.prototype.getMesh64=function(){return new ze(this._GetMesh64JS())},t.ManifoldError=function(N,...Z){const ae=t._ErrorMessage(t.status[N]?.value??-1),Te=Error.apply(this,[ae,...Z]);Te.name=this.name="ManifoldError",this.message=Te.message,this.stack=Te.stack,this.code=N},t.ManifoldError.prototype=Object.create(Error.prototype,{constructor:{value:t.ManifoldError,writable:!0,configurable:!0}}),t.CrossSection=function(L){const N=H(L),Z=new pe(N);return N.delete(),Z},t.CrossSection.evenOdd=function(L){const N=H(L),Z=t._EvenOdd(N);return N.delete(),Z},t.CrossSection.nonZero=function(L){const N=H(L),Z=t._NonZero(N);return N.delete(),Z},t.CrossSection.square=function(...L){let N;L.length==0?N={x:1,y:1}:typeof L[0]=="number"?N={x:L[0],y:L[0]}:N=X(L);const Z=L[1]||!1;return t._Square(N,Z)},t.CrossSection.circle=function(L,N=0){return t._Circle(L,f(N))};function qe(L){return function(...N){N.length==1&&(N=N[0]);const Z=new t.Vector_crossSection;for(const Te of N)ge(Te,Ze=>Z.push_back(Ze));const ae=t["_crossSection"+L](Z);return Z.delete(),ae}}t.CrossSection.union=qe("UnionN"),t.CrossSection.difference=qe("DifferenceN"),t.CrossSection.intersection=qe("IntersectionN");function lt(L,N){c(L,N,Z=>Z instanceof Array?{x:Z[0],y:Z[1]}:Z)}t.CrossSection.hull=function(...L){L.length==1&&(L=L[0]);let N=new t.Vector_vec2;for(const ae of L)if(ae instanceof pe)t._crossSectionCollectVertices(N,ae);else if(ae instanceof Array&&ae.length==2&&typeof ae[0]=="number")N.push_back({x:ae[0],y:ae[1]});else if(ae.x)N.push_back(ae);else{const Ze=ae[0].length==2&&typeof ae[0][0]=="number"||ae[0].x?[ae]:ae;for(const bt of Ze)lt(N,bt)}const Z=t._crossSectionHullPoints(N);return N.delete(),Z},t.CrossSection.prototype=Object.create(pe.prototype),Object.defineProperty(t.CrossSection,Symbol.hasInstance,{get:()=>L=>L instanceof pe});const Ct=t.Manifold;t.Manifold=function(L){return U(L?.[Xe]?t._FromMesh64(L):t._FromMesh(L))},t.Manifold.tetrahedron=function(){return t._Tetrahedron()},t.Manifold.cube=function(...L){let N;L.length==0?N={x:1,y:1,z:1}:typeof L[0]=="number"?N={x:L[0],y:L[0],z:L[0]}:N=J(L);const Z=L[1]||!1;return t._Cube(N,Z)},t.Manifold.cylinder=function(L,N,Z=-1,ae=0,Te=!1){return t._Cylinder(L,N,Z,f(ae),Te)},t.Manifold.sphere=function(L,N=0){return t._Sphere(L,f(N))},t.Manifold.extrude=function(L,N,Z=0,ae=0,Te=[1,1],Ze=!1){return ge(L,bt=>bt.extrude(N,Z,ae,Te,Ze))},t.Manifold.revolve=function(L,N=0,Z=360){return ge(L,ae=>ae.revolve(N,Z))},t.Manifold.reserveIDs=function(L){return t._ReserveIDs(L)};function Pt(L){return function(...N){N.length==1&&(N=N[0]);const Z=new t.Vector_manifold;for(const Te of N)Z.push_back(Te);const ae=t["_manifold"+L+"N"](Z);return Z.delete(),ae}}t.Manifold.union=Pt("Union"),t.Manifold.difference=Pt("Difference"),t.Manifold.intersection=Pt("Intersection");function tn(L,N,Z){return T(ae=>function(Te){if(ae.failed)return N-1;try{const Ze=Je(Te,"double"),bt=Je(Te+8,"double"),rt=Je(Te+16,"double");return L([Ze,bt,rt])}catch(Ze){return ae.failed=!0,ae.failure=Ze,N-1}},"di",Z)}t.Manifold.levelSet=function(L,N,Z,ae=0,Te=-1){const Ze={min:{x:N.min[0],y:N.min[1],z:N.min[2]},max:{x:N.max[0],y:N.max[1],z:N.max[2]}};return tn(L,ae,bt=>t._LevelSet(bt,Ze,Z,ae,Te))},t.ExecutionContext.prototype.fromMesh=function(L){return L?.[Xe]?this._FromMesh64(L):this._FromMesh(L)},t.ExecutionContext.prototype.levelSet=function(L,N,Z,ae=0,Te=-1){const Ze={min:{x:N.min[0],y:N.min[1],z:N.min[2]},max:{x:N.max[0],y:N.max[1],z:N.max[2]}};return tn(L,ae,bt=>this._LevelSet(bt,Ze,Z,ae,Te))};function jt(L,N){c(L,N,Z=>Z instanceof Array?{x:Z[0],y:Z[1],z:Z[2]}:Z)}t.Manifold.hull=function(...L){L.length==1&&(L=L[0]);const N=[],Z=new t.Vector_vec3;for(const rt of L)rt instanceof Ct?N.push(rt):rt instanceof Array&&rt.length==3&&typeof rt[0]=="number"?Z.push_back({x:rt[0],y:rt[1],z:rt[2]}):rt.x?Z.push_back(rt):jt(Z,rt);if(Z.size()==0){const rt=new t.Vector_manifold;c(rt,N),Z.delete();const Tn=t._manifoldHull(rt);return rt.delete(),Tn}let ae=1/0;for(const rt of N)t._manifoldCollectVertices(Z,rt),rt.isEmpty()||(ae=Math.min(ae,rt.getPrecision()));const Te=t._manifoldHullPoints(Z);if(Z.delete(),!isFinite(ae)||ae==Te.getPrecision())return Te;const Ze=Te.withPrecision(ae);Te.delete();const bt=Ze.hull();return Ze.delete(),bt},t.Manifold.prototype=Object.create(Ct.prototype),Object.defineProperty(t.Manifold,Symbol.hasInstance,{get:()=>L=>L instanceof Ct}),t.triangulate=function(L,N=-1,Z=!0){const ae=H(L),Te=t._Triangulate(ae,N,Z);ae.delete();const Ze=S(Te,bt=>[bt[0],bt[1],bt[2]]);return Te.delete(),Ze}};var u=import.meta.url,l="";function h(c){return t.locateFile?t.locateFile(c,l):l+c}var p,m;if(s){var v=a("node:fs");u.startsWith("file:")&&(l=a("node:path").dirname(a("node:url").fileURLToPath(u))+"/"),m=c=>{c=d(c)?new URL(c):c;var f=v.readFileSync(c);return f},p=async(c,f=!0)=>{c=d(c)?new URL(c):c;var S=v.readFileSync(c,f?void 0:"utf8");return S},process.argv.length>1&&process.argv[1].replace(/\\/g,"/"),process.argv.slice(2)}else if(n||r){try{l=new URL(".",u).href}catch{}r&&(m=c=>{var f=new XMLHttpRequest;return f.open("GET",c,!1),f.responseType="arraybuffer",f.send(null),new Uint8Array(f.response)}),p=async c=>{if(d(c))return new Promise((S,T)=>{var U=new XMLHttpRequest;U.open("GET",c,!0),U.responseType="arraybuffer",U.onload=()=>{if(U.status==200||U.status==0&&U.response){S(U.response);return}T(U.status)},U.onerror=T,U.send(null)});var f=await fetch(c,{credentials:"same-origin"});if(f.ok)return f.arrayBuffer();throw new Error(f.status+" : "+f.url)}}console.log.bind(console);var y=console.error.bind(console),b,g=!1,d=c=>c.startsWith("file://"),R,C,A,P,D,I,G,x,E,B,j,K,ie=!1;function oe(){var c=wr.buffer;A=new Int8Array(c),D=new Int16Array(c),P=new Uint8Array(c),I=new Uint16Array(c),G=new Int32Array(c),x=new Uint32Array(c),E=new Float32Array(c),B=new Float64Array(c),j=new BigInt64Array(c),K=new BigUint64Array(c)}function Q(){if(t.preRun)for(typeof t.preRun=="function"&&(t.preRun=[t.preRun]);t.preRun.length;)Vt(t.preRun.shift());Ne(ht)}function te(){ie=!0,ci.K()}function ue(){if(t.postRun)for(typeof t.postRun=="function"&&(t.postRun=[t.postRun]);t.postRun.length;)Ve(t.postRun.shift());Ne(Qe)}function Ae(c){t.onAbort?.(c),c="Aborted("+c+")",y(c),g=!0,c+=". Build with -sASSERTIONS for more info.";var f=new WebAssembly.RuntimeError(c);throw C?.(f),f}var Ee;function Le(){return t.locateFile?h("manifold.wasm"):new URL(""+new URL("manifold.wasm",import.meta.url).href,import.meta.url).href}function at(c){if(c==Ee&&b)return new Uint8Array(b);if(m)return m(c);throw"both async and sync fetching of the wasm failed"}async function it(c){if(!b)try{var f=await p(c);return new Uint8Array(f)}catch{}return at(c)}async function Ut(c,f){try{var S=await it(c),T=await WebAssembly.instantiate(S,f);return T}catch(U){y(`failed to asynchronously prepare wasm: ${U}`),Ae(U)}}async function Dt(c,f,S){if(!c&&!d(f)&&!s)try{var T=fetch(f,{credentials:"same-origin"}),U=await WebAssembly.instantiateStreaming(T,S);return U}catch(k){y(`wasm streaming compile failed: ${k}`),y("falling back to ArrayBuffer instantiation")}return Ut(f,S)}function re(){var c={a:Kc};return c}async function fe(){function c(k,H){return ci=k.exports,ci=jc(ci),Yc(ci),oe(),ci}function f(k){return c(k.instance)}var S=re();if(t.instantiateWasm)return new Promise((k,H)=>{t.instantiateWasm(S,(X,J)=>{k(c(X))})});Ee??=Le();var T=await Dt(b,Ee,S),U=f(T);return U}var Ne=c=>{for(;c.length>0;)c.shift()(t)},Qe=[],Ve=c=>Qe.push(c),ht=[],Vt=c=>ht.push(c);function Je(c,f="i8"){switch(f.endsWith("*")&&(f="*"),f){case"i1":return A[c>>>0];case"i8":return A[c>>>0];case"i16":return D[c>>>1>>>0];case"i32":return G[c>>>2>>>0];case"i64":return j[c>>>3>>>0];case"float":return E[c>>>2>>>0];case"double":return B[c>>>3>>>0];case"*":return x[c>>>2>>>0];default:Ae(`invalid type for getValue: ${f}`)}}function ut(c,f,S="i8"){switch(S.endsWith("*")&&(S="*"),S){case"i1":A[c>>>0]=f;break;case"i8":A[c>>>0]=f;break;case"i16":D[c>>>1>>>0]=f;break;case"i32":G[c>>>2>>>0]=f;break;case"i64":j[c>>>3>>>0]=BigInt(f);break;case"float":E[c>>>2>>>0]=f;break;case"double":B[c>>>3>>>0]=f;break;case"*":x[c>>>2>>>0]=f;break;default:Ae(`invalid type for setValue: ${S}`)}}class Tt{constructor(f){this.excPtr=f,this.ptr=f-24}set_type(f){x[this.ptr+4>>>2>>>0]=f}get_type(){return x[this.ptr+4>>>2>>>0]}set_destructor(f){x[this.ptr+8>>>2>>>0]=f}get_destructor(){return x[this.ptr+8>>>2>>>0]}set_caught(f){f=f?1:0,A[this.ptr+12>>>0]=f}get_caught(){return A[this.ptr+12>>>0]!=0}set_rethrown(f){f=f?1:0,A[this.ptr+13>>>0]=f}get_rethrown(){return A[this.ptr+13>>>0]!=0}init(f,S){this.set_adjusted_ptr(0),this.set_type(f),this.set_destructor(S)}set_adjusted_ptr(f){x[this.ptr+16>>>2>>>0]=f}get_adjusted_ptr(){return x[this.ptr+16>>>2>>>0]}}var nt=0;function Ft(c,f,S){c>>>=0,f>>>=0,S>>>=0;var T=new Tt(c);throw T.init(f,S),nt=c,nt}var F=()=>Ae(""),It={},mt=c=>{for(;c.length;){var f=c.pop(),S=c.pop();S(f)}};function gt(c){return this.fromWireType(x[c>>>2>>>0])}var Oe={},w={},_={},V=class extends Error{constructor(f){super(f),this.name="InternalError"}},ne=c=>{throw new V(c)},se=(c,f,S)=>{c.forEach(X=>_[X]=f);function T(X){var J=S(X);J.length!==c.length&&ne("Mismatched type converter count");for(var he=0;he<c.length;++he)ce(c[he],J[he])}var U=new Array(f.length),k=[],H=0;for(let[X,J]of f.entries())w.hasOwnProperty(J)?U[X]=w[J]:(k.push(J),Oe.hasOwnProperty(J)||(Oe[J]=[]),Oe[J].push(()=>{U[X]=w[J],++H,H===k.length&&T(U)}));k.length===0&&T(U)},ee=function(c){c>>>=0;var f=It[c];delete It[c];var S=f.rawConstructor,T=f.rawDestructor,U=f.fields,k=U.map(H=>H.getterReturnType).concat(U.map(H=>H.setterArgumentType));se([c],k,H=>{var X={};for(var[J,he]of U.entries()){const pe=H[J],ge=he.getter,Ce=he.getterContext,Ue=H[J+U.length],ze=he.setter,Xe=he.setterContext;X[he.fieldName]={read:qe=>pe.fromWireType(ge(Ce,qe)),write:(qe,lt)=>{var Ct=[];ze(Xe,qe,Ue.toWireType(Ct,lt)),mt(Ct)},optional:pe.optional}}return[{name:f.name,fromWireType:pe=>{var ge={};for(var Ce in X)ge[Ce]=X[Ce].read(pe);return T(pe),ge},toWireType:(pe,ge)=>{for(var Ce in X)if(!(Ce in ge)&&!X[Ce].optional)throw new TypeError(`Missing field: "${Ce}"`);var Ue=S();for(Ce in X)X[Ce].write(Ue,ge[Ce]);return pe!==null&&pe.push(T,Ue),Ue},readValueFromPointer:gt,destructorFunction:T}]})},Me=c=>{c>>>=0;for(var f="";;){var S=P[c++>>>0];if(!S)return f;f+=String.fromCharCode(S)}},ve=class extends Error{constructor(f){super(f),this.name="BindingError"}},me=c=>{throw new ve(c)};function Ye(c,f,S={}){var T=f.name;if(c||me(`type "${T}" must have a positive integer typeid pointer`),w.hasOwnProperty(c)){if(S.ignoreDuplicateRegistrations)return;me(`Cannot register type '${T}' twice`)}if(w[c]=f,delete _[c],Oe.hasOwnProperty(c)){var U=Oe[c];delete Oe[c],U.forEach(k=>k())}}function ce(c,f,S={}){return Ye(c,f,S)}var Se=(c,f,S)=>{switch(f){case 1:return S?T=>A[T>>>0]:T=>P[T>>>0];case 2:return S?T=>D[T>>>1>>>0]:T=>I[T>>>1>>>0];case 4:return S?T=>G[T>>>2>>>0]:T=>x[T>>>2>>>0];case 8:return S?T=>j[T>>>3>>>0]:T=>K[T>>>3>>>0];default:throw new TypeError(`invalid integer width (${f}): ${c}`)}},Be=function(c,f,S,T,U){c>>>=0,f>>>=0,S>>>=0,f=Me(f);const k=T===0n;let H=X=>X;if(k){const X=S*8;H=J=>BigInt.asUintN(X,J),U=H(U)}ce(c,{name:f,fromWireType:H,toWireType:(X,J)=>(typeof J=="number"&&(J=BigInt(J)),J),readValueFromPointer:Se(f,S,!k),destructorFunction:null})};function Ge(c,f,S,T){c>>>=0,f>>>=0,f=Me(f),ce(c,{name:f,fromWireType:function(U){return!!U},toWireType:function(U,k){return k?S:T},readValueFromPointer:function(U){return this.fromWireType(P[U>>>0])},destructorFunction:null})}var ye=c=>({count:c.count,deleteScheduled:c.deleteScheduled,preservePointerOnDelete:c.preservePointerOnDelete,ptr:c.ptr,ptrType:c.ptrType,smartPtr:c.smartPtr,smartPtrType:c.smartPtrType}),tt=c=>{function f(S){return S.$$.ptrType.registeredClass.name}me(f(c)+" instance already deleted")},O=!1,we=c=>{},_e=c=>{c.smartPtr?c.smartPtrType.rawDestructor(c.smartPtr):c.ptrType.registeredClass.rawDestructor(c.ptr)},De=c=>{c.count.value-=1;var f=c.count.value===0;f&&_e(c)},de=c=>globalThis.FinalizationRegistry?(O=new FinalizationRegistry(f=>{De(f.$$)}),de=f=>{var S=f.$$,T=!!S.smartPtr;if(T){var U={$$:S};O.register(f,U,f)}return f},we=f=>O.unregister(f),de(c)):(de=f=>f,c),le=()=>{let c=xe.prototype;Object.assign(c,{isAliasOf(S){if(!(this instanceof xe)||!(S instanceof xe))return!1;var T=this.$$.ptrType.registeredClass,U=this.$$.ptr;S.$$=S.$$;for(var k=S.$$.ptrType.registeredClass,H=S.$$.ptr;T.baseClass;)U=T.upcast(U),T=T.baseClass;for(;k.baseClass;)H=k.upcast(H),k=k.baseClass;return T===k&&U===H},clone(){if(this.$$.ptr||tt(this),this.$$.preservePointerOnDelete)return this.$$.count.value+=1,this;var S=de(Object.create(Object.getPrototypeOf(this),{$$:{value:ye(this.$$)}}));return S.$$.count.value+=1,S.$$.deleteScheduled=!1,S},delete(){this.$$.ptr||tt(this),this.$$.deleteScheduled&&!this.$$.preservePointerOnDelete&&me("Object already scheduled for deletion"),we(this),De(this.$$),this.$$.preservePointerOnDelete||(this.$$.smartPtr=void 0,this.$$.ptr=void 0)},isDeleted(){return!this.$$.ptr},deleteLater(){return this.$$.ptr||tt(this),this.$$.deleteScheduled&&!this.$$.preservePointerOnDelete&&me("Object already scheduled for deletion"),this.$$.deleteScheduled=!0,this}});const f=Symbol.dispose;f&&(c[f]=c.delete)};function xe(){}var Ke=(c,f)=>Object.defineProperty(f,"name",{value:c}),At={},vt=(c,f,S)=>{if(c[f].overloadTable===void 0){var T=c[f];c[f]=function(...U){return c[f].overloadTable.hasOwnProperty(U.length)||me(`Function '${S}' called with an invalid number of arguments (${U.length}) - expects one of (${c[f].overloadTable})!`),c[f].overloadTable[U.length].apply(this,U)},c[f].overloadTable=[],c[f].overloadTable[T.argCount]=T}},$t=(c,f,S)=>{t.hasOwnProperty(c)?((S===void 0||t[c].overloadTable!==void 0&&t[c].overloadTable[S]!==void 0)&&me(`Cannot register public name '${c}' twice`),vt(t,c,c),t[c].overloadTable.hasOwnProperty(S)&&me(`Cannot register multiple overloads of a function with the same number of arguments (${S})!`),t[c].overloadTable[S]=f):(t[c]=f,t[c].argCount=S)},vn=48,ds=57,Mr=c=>{c=c.replace(/[^a-zA-Z0-9_]/g,"$");var f=c.charCodeAt(0);return f>=vn&&f<=ds?`_${c}`:c};function Zi(c,f,S,T,U,k,H,X){this.name=c,this.constructor=f,this.instancePrototype=S,this.rawDestructor=T,this.baseClass=U,this.getActualType=k,this.upcast=H,this.downcast=X,this.pureVirtualFunctions=[]}var Ji=(c,f,S)=>{for(;f!==S;)f.upcast||me(`Expected null or instance of ${S.name}, got an instance of ${f.name}`),c=f.upcast(c),f=f.baseClass;return c},yi=c=>{if(c===null)return"null";var f=typeof c;return f==="object"||f==="array"||f==="function"?c.toString():""+c};function Sr(c,f){if(f===null)return this.isReference&&me(`null is not a valid ${this.name}`),0;f.$$||me(`Cannot pass "${yi(f)}" as a ${this.name}`),f.$$.ptr||me(`Cannot pass deleted object as a pointer of type ${this.name}`);var S=f.$$.ptrType.registeredClass,T=Ji(f.$$.ptr,S,this.registeredClass);return T}function Nn(c,f){var S;if(f===null)return this.isReference&&me(`null is not a valid ${this.name}`),this.isSmartPointer?(S=this.rawConstructor(),c!==null&&c.push(this.rawDestructor,S),S):0;(!f||!f.$$)&&me(`Cannot pass "${yi(f)}" as a ${this.name}`),f.$$.ptr||me(`Cannot pass deleted object as a pointer of type ${this.name}`),!this.isConst&&f.$$.ptrType.isConst&&me(`Cannot convert argument of type ${f.$$.smartPtrType?f.$$.smartPtrType.name:f.$$.ptrType.name} to parameter type ${this.name}`);var T=f.$$.ptrType.registeredClass;if(S=Ji(f.$$.ptr,T,this.registeredClass),this.isSmartPointer)switch(f.$$.smartPtr===void 0&&me("Passing raw pointer to smart pointer is illegal"),this.sharingPolicy){case 0:f.$$.smartPtrType===this?S=f.$$.smartPtr:me(`Cannot convert argument of type ${f.$$.smartPtrType?f.$$.smartPtrType.name:f.$$.ptrType.name} to parameter type ${this.name}`);break;case 1:S=f.$$.smartPtr;break;case 2:if(f.$$.smartPtrType===this)S=f.$$.smartPtr;else{var U=f.clone();S=this.rawShare(S,je.toHandle(()=>U.delete())),c!==null&&c.push(this.rawDestructor,S)}break;default:me("Unsupported sharing policy")}return S}function Qi(c,f){if(f===null)return this.isReference&&me(`null is not a valid ${this.name}`),0;f.$$||me(`Cannot pass "${yi(f)}" as a ${this.name}`),f.$$.ptr||me(`Cannot pass deleted object as a pointer of type ${this.name}`),f.$$.ptrType.isConst&&me(`Cannot convert argument of type ${f.$$.ptrType.name} to parameter type ${this.name}`);var S=f.$$.ptrType.registeredClass,T=Ji(f.$$.ptr,S,this.registeredClass);return T}var er=(c,f,S)=>{if(f===S)return c;if(S.baseClass===void 0)return null;var T=er(c,f,S.baseClass);return T===null?null:S.downcast(T)},yr={},Ei=(c,f)=>{for(f===void 0&&me("ptr should not be undefined");c.baseClass;)f=c.upcast(f),c=c.baseClass;return f},Er=(c,f)=>(f=Ei(c,f),yr[f]),On=(c,f)=>{(!f.ptrType||!f.ptr)&&ne("makeClassHandle requires ptr and ptrType");var S=!!f.smartPtrType,T=!!f.smartPtr;return S!==T&&ne("Both smartPtrType and smartPtr must be specified"),f.count={value:1},de(Object.create(c,{$$:{value:f,writable:!0}}))};function br(c){var f=this.getPointee(c);if(!f)return this.destructor(c),null;var S=Er(this.registeredClass,f);if(S!==void 0){if(S.$$.count.value===0)return S.$$.ptr=f,S.$$.smartPtr=c,S.clone();var T=S.clone();return this.destructor(c),T}function U(){return this.isSmartPointer?On(this.registeredClass.instancePrototype,{ptrType:this.pointeeType,ptr:f,smartPtrType:this,smartPtr:c}):On(this.registeredClass.instancePrototype,{ptrType:this,ptr:c})}var k=this.registeredClass.getActualType(f),H=At[k];if(!H)return U.call(this);var X;this.isConst?X=H.constPointerType:X=H.pointerType;var J=er(f,this.registeredClass,X.registeredClass);return J===null?U.call(this):this.isSmartPointer?On(X.registeredClass.instancePrototype,{ptrType:X,ptr:J,smartPtrType:this,smartPtr:c}):On(X.registeredClass.instancePrototype,{ptrType:X,ptr:J})}var Tr=()=>{Object.assign(bi.prototype,{getPointee(c){return this.rawGetPointee&&(c=this.rawGetPointee(c)),c},destructor(c){this.rawDestructor?.(c)},readValueFromPointer:gt,fromWireType:br})};function bi(c,f,S,T,U,k,H,X,J,he,pe){this.name=c,this.registeredClass=f,this.isReference=S,this.isConst=T,this.isSmartPointer=U,this.pointeeType=k,this.sharingPolicy=H,this.rawGetPointee=X,this.rawConstructor=J,this.rawShare=he,this.rawDestructor=pe,!U&&f.baseClass===void 0?T?(this.toWireType=Sr,this.destructorFunction=null):(this.toWireType=Qi,this.destructorFunction=null):this.toWireType=Nn}var Ar=(c,f,S)=>{t.hasOwnProperty(c)||ne("Replacing nonexistent public symbol"),t[c].overloadTable!==void 0&&S!==void 0?t[c].overloadTable[S]=f:(t[c]=f,t[c].argCount=S)},Ti=c=>nr.get(c),ps=(c,f,S=[],T=!1)=>{var U=Ti(f),k=U(...S);function H(X){return c[0]=="p"?X>>>0:X}return H(k)},ms=(c,f,S=!1)=>(...T)=>ps(c,f,T,S),cn=(c,f,S=!1)=>{c=Me(c);function T(){if(c.includes("p"))return ms(c,f,S);var k=Ti(f);return k}var U=T();return typeof U!="function"&&me(`unknown function pointer with signature ${c}: ${f}`),U};class M extends Error{}var z=c=>{var f=Mo(c),S=Me(f);return Zn(f),S},Y=(c,f)=>{var S=[],T={};function U(k){if(!T[k]&&!w[k]){if(_[k]){_[k].forEach(U);return}S.push(k),T[k]=!0}}throw f.forEach(U),new M(`${c}: `+S.map(z).join([", "]))};function q(c,f,S,T,U,k,H,X,J,he,pe,ge,Ce){c>>>=0,f>>>=0,S>>>=0,T>>>=0,U>>>=0,k>>>=0,H>>>=0,X>>>=0,J>>>=0,he>>>=0,pe>>>=0,ge>>>=0,Ce>>>=0,pe=Me(pe),k=cn(U,k),X&&=cn(H,X),he&&=cn(J,he),Ce=cn(ge,Ce);var Ue=Mr(pe);$t(Ue,function(){Y(`Cannot construct ${pe} due to unbound types`,[T])}),se([c,f,S],T?[T]:[],ze=>{ze=ze[0];var Xe,qe;T?(Xe=ze.registeredClass,qe=Xe.instancePrototype):qe=xe.prototype;var lt=Ke(pe,function(...N){if(Object.getPrototypeOf(this)!==Ct)throw new ve(`Use 'new' to construct ${pe}`);if(Pt.constructor_body===void 0)throw new ve(`${pe} has no accessible constructor`);var Z=Pt.constructor_body[N.length];if(Z===void 0)throw new ve(`Tried to invoke ctor of ${pe} with invalid number of parameters (${N.length}) - expected (${Object.keys(Pt.constructor_body).toString()}) parameters instead!`);return Z.apply(this,N)}),Ct=Object.create(qe,{constructor:{value:lt}});lt.prototype=Ct;var Pt=new Zi(pe,lt,Ct,Ce,Xe,k,X,he);Pt.baseClass&&(Pt.baseClass.__derivedClasses??=[],Pt.baseClass.__derivedClasses.push(Pt));var tn=new bi(pe,Pt,!0,!1,!1),jt=new bi(pe+"*",Pt,!1,!1,!1),L=new bi(pe+" const*",Pt,!1,!0,!1);return At[c]={pointerType:jt,constPointerType:L},Ar(Ue,lt),[tn,jt,L]})}var W=(c,f)=>{for(var S=[],T=0;T<c;T++)S.push(x[f+T*4>>>2>>>0]);return S};function be(c){for(var f=1;f<c.length;++f)if(c[f]!==null&&c[f].destructorFunction===void 0)return!0;return!1}var Ie={ftfn:function(f,S,T,U,k,H,X,J){return function(he){var pe=J(null,he),ge=T(U,pe),Ce=H(ge);return Ce}},ftf:function(f,S,T,U,k,H,X){return function(){var J=T(U),he=H(J);return he}},tffn:function(f,S,T,U,k,H,X){return function(){var J=X(null,this);T(U,J)}},ttfn:function(f,S,T,U,k,H,X){return function(){var J=X(null,this),he=T(U,J),pe=H(he);return pe}},ttfnn:function(f,S,T,U,k,H,X,J){return function(he){var pe=X(null,this),ge=J(null,he),Ce=T(U,pe,ge),Ue=H(Ce);return Ue}},ttfnntnnn:function(f,S,T,U,k,H,X,J,he,pe,ge,Ce,Ue){return function(ze,Xe,qe,lt,Ct){var Pt=X(null,this),tn=J(null,ze),jt=he(null,Xe),L=pe(null,qe),N=ge(null,lt),Z=Ce(null,Ct),ae=T(U,Pt,tn,jt,L,N,Z);Ue(jt);var Te=H(ae);return Te}},tffnt:function(f,S,T,U,k,H,X,J,he){return function(pe){var ge=X(null,this),Ce=J(null,pe);T(U,ge,Ce),he(Ce)}},tffnnt:function(f,S,T,U,k,H,X,J,he,pe){return function(ge,Ce){var Ue=X(null,this),ze=J(null,ge),Xe=he(null,Ce);T(U,Ue,ze,Xe),pe(Xe)}},ttfnnt:function(f,S,T,U,k,H,X,J,he,pe){return function(ge,Ce){var Ue=X(null,this),ze=J(null,ge),Xe=he(null,Ce),qe=T(U,Ue,ze,Xe);pe(Xe);var lt=H(qe);return lt}},tffnn:function(f,S,T,U,k,H,X,J){return function(he){var pe=X(null,this),ge=J(null,he);T(U,pe,ge)}},tffnnn:function(f,S,T,U,k,H,X,J,he){return function(pe,ge){var Ce=X(null,this),Ue=J(null,pe),ze=he(null,ge);T(U,Ce,Ue,ze)}},ttfnnn:function(f,S,T,U,k,H,X,J,he){return function(pe,ge){var Ce=X(null,this),Ue=J(null,pe),ze=he(null,ge),Xe=T(U,Ce,Ue,ze),qe=H(Xe);return qe}},ttfnt:function(f,S,T,U,k,H,X,J,he){return function(pe){var ge=X(null,this),Ce=J(null,pe),Ue=T(U,ge,Ce);he(Ce);var ze=H(Ue);return ze}},ttfnnnnt:function(f,S,T,U,k,H,X,J,he,pe,ge,Ce){return function(Ue,ze,Xe,qe){var lt=X(null,this),Ct=J(null,Ue),Pt=he(null,ze),tn=pe(null,Xe),jt=ge(null,qe),L=T(U,lt,Ct,Pt,tn,jt);Ce(jt);var N=H(L);return N}},ftftn:function(f,S,T,U,k,H,X,J,he,pe){return function(ge,Ce){var Ue=J(null,ge),ze=he(null,Ce),Xe=T(U,Ue,ze);pe(Ue);var qe=H(Xe);return qe}},ftfnt:function(f,S,T,U,k,H,X,J,he,pe){return function(ge,Ce){var Ue=J(null,ge),ze=he(null,Ce),Xe=T(U,Ue,ze);pe(ze);var qe=H(Xe);return qe}},fffnn:function(f,S,T,U,k,H,X,J,he){return function(pe,ge){var Ce=J(null,pe),Ue=he(null,ge);T(U,Ce,Ue)}},ttfntn:function(f,S,T,U,k,H,X,J,he,pe){return function(ge,Ce){var Ue=X(null,this),ze=J(null,ge),Xe=he(null,Ce),qe=T(U,Ue,ze,Xe);pe(ze);var lt=H(qe);return lt}},ttfnnnn:function(f,S,T,U,k,H,X,J,he,pe){return function(ge,Ce,Ue){var ze=X(null,this),Xe=J(null,ge),qe=he(null,Ce),lt=pe(null,Ue),Ct=T(U,ze,Xe,qe,lt),Pt=H(Ct);return Pt}},ttfntt:function(f,S,T,U,k,H,X,J,he,pe,ge){return function(Ce,Ue){var ze=X(null,this),Xe=J(null,Ce),qe=he(null,Ue),lt=T(U,ze,Xe,qe);pe(Xe),ge(qe);var Ct=H(lt);return Ct}},ftfnnntn:function(f,S,T,U,k,H,X,J,he,pe,ge,Ce,Ue){return function(ze,Xe,qe,lt,Ct){var Pt=J(null,ze),tn=he(null,Xe),jt=pe(null,qe),L=ge(null,lt),N=Ce(null,Ct),Z=T(U,Pt,tn,jt,L,N);Ue(L);var ae=H(Z);return ae}},ftfnnnnt:function(f,S,T,U,k,H,X,J,he,pe,ge,Ce,Ue){return function(ze,Xe,qe,lt,Ct){var Pt=J(null,ze),tn=he(null,Xe),jt=pe(null,qe),L=ge(null,lt),N=Ce(null,Ct),Z=T(U,Pt,tn,jt,L,N);Ue(N);var ae=H(Z);return ae}},ftfnnn:function(f,S,T,U,k,H,X,J,he,pe){return function(ge,Ce,Ue){var ze=J(null,ge),Xe=he(null,Ce),qe=pe(null,Ue),lt=T(U,ze,Xe,qe),Ct=H(lt);return Ct}},ftfntn:function(f,S,T,U,k,H,X,J,he,pe,ge){return function(Ce,Ue,ze){var Xe=J(null,Ce),qe=he(null,Ue),lt=pe(null,ze),Ct=T(U,Xe,qe,lt);ge(qe);var Pt=H(Ct);return Pt}},ftfntnnn:function(f,S,T,U,k,H,X,J,he,pe,ge,Ce,Ue){return function(ze,Xe,qe,lt,Ct){var Pt=J(null,ze),tn=he(null,Xe),jt=pe(null,qe),L=ge(null,lt),N=Ce(null,Ct),Z=T(U,Pt,tn,jt,L,N);Ue(tn);var ae=H(Z);return ae}},ftfnn:function(f,S,T,U,k,H,X,J,he){return function(pe,ge){var Ce=J(null,pe),Ue=he(null,ge),ze=T(U,Ce,Ue),Xe=H(ze);return Xe}},fff:function(f,S,T,U,k,H,X){return function(){T(U)}}};function Re(c,f,S,T){const U=[f?"t":"f",S?"t":"f",T?"t":"f"];for(let k=f?1:2;k<c.length;++k){const H=c[k];let X="";H.destructorFunction===void 0?X="u":H.destructorFunction===null?X="n":X="t",U.push(X)}return U.join("")}function Fe(c,f,S,T,U,k){var H=f.length;H<2&&me("argTypes array size mismatch! Must at least get return value and 'this' types!");for(var X=f[1]!==null&&S!==null,J=be(f),he=!f[0].isVoid,pe=f[0],ge=f[1],Ce=[c,me,T,U,mt,pe.fromWireType.bind(pe),ge?.toWireType.bind(ge)],Ue=2;Ue<H;++Ue){var ze=f[Ue];Ce.push(ze.toWireType.bind(ze))}if(!J)for(var Ue=X?1:2;Ue<f.length;++Ue)f[Ue].destructorFunction!==null&&Ce.push(f[Ue].destructorFunction);var Xe=Re(f,X,he,k),qe=Ie[Xe](...Ce);return Ke(c,qe)}var He=function(c,f,S,T,U,k){c>>>=0,S>>>=0,T>>>=0,U>>>=0,k>>>=0;var H=W(f,S);U=cn(T,U),se([],[c],X=>{X=X[0];var J=`constructor ${X.name}`;if(X.registeredClass.constructor_body===void 0&&(X.registeredClass.constructor_body=[]),X.registeredClass.constructor_body[f-1]!==void 0)throw new ve(`Cannot register multiple constructors with identical number of parameters (${f-1}) for class '${X.name}'! Overload resolution is currently only performed using the parameter count, not actual type info!`);return X.registeredClass.constructor_body[f-1]=()=>{Y(`Cannot construct ${X.name} due to unbound types`,H)},se([],H,he=>(he.splice(1,0,null),X.registeredClass.constructor_body[f-1]=Fe(J,he,null,U,k),[])),[]})},$e=c=>{c=c.trim();const f=c.indexOf("(");return f===-1?c:c.slice(0,f)},We=function(c,f,S,T,U,k,H,X,J,he){c>>>=0,f>>>=0,T>>>=0,U>>>=0,k>>>=0,H>>>=0;var pe=W(S,T);f=Me(f),f=$e(f),k=cn(U,k,J),se([],[c],ge=>{ge=ge[0];var Ce=`${ge.name}.${f}`;f.startsWith("@@")&&(f=Symbol[f.substring(2)]),X&&ge.registeredClass.pureVirtualFunctions.push(f);function Ue(){Y(`Cannot call ${Ce} due to unbound types`,pe)}var ze=ge.registeredClass.instancePrototype,Xe=ze[f];return Xe===void 0||Xe.overloadTable===void 0&&Xe.className!==ge.name&&Xe.argCount===S-2?(Ue.argCount=S-2,Ue.className=ge.name,ze[f]=Ue):(vt(ze,f,Ce),ze[f].overloadTable[S-2]=Ue),se([],pe,qe=>{var lt=Fe(Ce,qe,ge,k,H,J);return ze[f].overloadTable===void 0?(lt.argCount=S-2,ze[f]=lt):ze[f].overloadTable[S-2]=lt,[]}),[]})},ot=function(c,f,S){c>>>=0,f>>>=0,c=Me(c),se([],[f],T=>(T=T[0],t[c]=T.fromWireType(S),[]))},Mt=[],Et=[0,1,,1,null,1,!0,1,!1,1];function Lt(c){c>>>=0,c>9&&--Et[c+1]===0&&(Et[c]=void 0,Mt.push(c))}var je={toValue:c=>(c||me(`Cannot use deleted val. handle = ${c}`),Et[c]),toHandle:c=>{switch(c){case void 0:return 2;case null:return 4;case!0:return 6;case!1:return 8;default:{const f=Mt.pop()||Et.length;return Et[f]=c,Et[f+1]=1,f}}}},ke={name:"emscripten::val",fromWireType:c=>{var f=je.toValue(c);return Lt(c),f},toWireType:(c,f)=>je.toHandle(f),readValueFromPointer:gt,destructorFunction:null};function St(c){return c>>>=0,ce(c,ke)}var ft=(c,f,S)=>{switch(f){case 1:return S?function(T){return this.fromWireType(A[T>>>0])}:function(T){return this.fromWireType(P[T>>>0])};case 2:return S?function(T){return this.fromWireType(D[T>>>1>>>0])}:function(T){return this.fromWireType(I[T>>>1>>>0])};case 4:return S?function(T){return this.fromWireType(G[T>>>2>>>0])}:function(T){return this.fromWireType(x[T>>>2>>>0])};default:throw new TypeError(`invalid integer width (${f}): ${c}`)}};function en(c){return c===0?"object":c===1?"number":"string"}function jn(c,f,S,T,U){c>>>=0,f>>>=0,S>>>=0,f=Me(f);const k=en(U);switch(k){case"object":{let he=function(){};he.values={},ce(c,{name:f,constructor:he,valueType:k,fromWireType:function(pe){return this.constructor.values[pe]},toWireType:(pe,ge)=>ge.value,readValueFromPointer:ft(f,S,T),destructorFunction:null}),$t(f,he);break}case"number":{var H={};ce(c,{name:f,keysMap:H,valueType:k,fromWireType:he=>he,toWireType:(he,pe)=>pe,readValueFromPointer:ft(f,S,T),destructorFunction:null}),$t(f,H),delete t[f].argCount;break}case"string":{var X={},J={},H={};ce(c,{name:f,valuesMap:X,reverseMap:J,keysMap:H,valueType:k,fromWireType:function(pe){return this.reverseMap[pe]},toWireType:function(pe,ge){return this.valuesMap[ge]},readValueFromPointer:ft(f,S,T),destructorFunction:null}),$t(f,H),delete t[f].argCount;break}}}var qt=(c,f)=>{var S=w[c];return S===void 0&&me(`${f} has unknown type ${z(c)}`),S};function oi(c,f,S){c>>>=0,f>>>=0;var T=qt(c,"enum");switch(f=Me(f),T.valueType){case"object":{var U=T.constructor,k=Object.create(T.constructor.prototype,{value:{value:S},constructor:{value:Ke(`${T.name}_${f}`,function(){})}});U.values[S]=k,U[f]=k;break}case"number":{T.keysMap[f]=S;break}case"string":{T.valuesMap[f]=S,T.reverseMap[S]=f,T.keysMap[f]=f;break}}}var Rt=(c,f)=>{switch(f){case 4:return function(S){return this.fromWireType(E[S>>>2>>>0])};case 8:return function(S){return this.fromWireType(B[S>>>3>>>0])};default:throw new TypeError(`invalid float width (${f}): ${c}`)}},Yt=function(c,f,S){c>>>=0,f>>>=0,S>>>=0,f=Me(f),ce(c,{name:f,fromWireType:T=>T,toWireType:(T,U)=>U,readValueFromPointer:Rt(f,S),destructorFunction:null})};function sn(c,f,S,T,U,k,H,X){c>>>=0,S>>>=0,T>>>=0,U>>>=0,k>>>=0;var J=W(f,S);c=Me(c),c=$e(c),U=cn(T,U,H),$t(c,function(){Y(`Cannot call ${c} due to unbound types`,J)},f-1),se([],J,he=>{var pe=[he[0],null].concat(he.slice(1));return Ar(c,Fe(c,pe,null,U,k,H),f-1),[]})}var Wt=function(c,f,S,T,U){c>>>=0,f>>>=0,S>>>=0,f=Me(f);const k=T===0;let H=J=>J;if(k){var X=32-8*S;H=J=>J<<X>>>X,U=H(U)}ce(c,{name:f,fromWireType:H,toWireType:(J,he)=>he,readValueFromPointer:Se(f,S,T!==0),destructorFunction:null})},Kt=(c,f,S)=>{const T=(U,k)=>{let H=0;return{next(){if(H>=U)return{done:!0};const X=H;return H++,{value:k(X),done:!1}},[Symbol.iterator](){return this}}};c[Symbol.iterator]||(c[Symbol.iterator]=function(){const U=this[f]();return T(U,k=>this[S](k))})},tr=function(c,f,S,T){c>>>=0,f>>>=0,S>>>=0,T>>>=0,S=Me(S),T=Me(T),se([],[c,f],U=>{const k=U[0];return Kt(k.registeredClass.instancePrototype,S,T),[]})};function Bn(c,f,S){c>>>=0,S>>>=0;var T=[Int8Array,Uint8Array,Int16Array,Uint16Array,Int32Array,Uint32Array,Float32Array,Float64Array,BigInt64Array,BigUint64Array],U=T[f];function k(H){var X=x[H>>>2>>>0],J=x[H+4>>>2>>>0];return new U(A.buffer,J,X)}S=Me(S),ce(c,{name:S,fromWireType:k,readValueFromPointer:k},{ignoreDuplicateRegistrations:!0})}var oc=Object.assign({optional:!0},ke);function lc(c,f){c>>>=0,ce(c,oc)}var cc=(c,f,S,T)=>{if(S>>>=0,!(T>0))return 0;for(var U=S,k=S+T-1,H=0;H<c.length;++H){var X=c.codePointAt(H);if(X<=127){if(S>=k)break;f[S++>>>0]=X}else if(X<=2047){if(S+1>=k)break;f[S++>>>0]=192|X>>6,f[S++>>>0]=128|X&63}else if(X<=65535){if(S+2>=k)break;f[S++>>>0]=224|X>>12,f[S++>>>0]=128|X>>6&63,f[S++>>>0]=128|X&63}else{if(S+3>=k)break;f[S++>>>0]=240|X>>18,f[S++>>>0]=128|X>>12&63,f[S++>>>0]=128|X>>6&63,f[S++>>>0]=128|X&63,H++}}return f[S>>>0]=0,S-U},uc=(c,f,S)=>cc(c,P,f,S),fc=c=>{for(var f=0,S=0;S<c.length;++S){var T=c.charCodeAt(S);T<=127?f++:T<=2047?f+=2:T>=55296&&T<=57343?(f+=4,++S):f+=3}return f},fo=globalThis.TextDecoder&&new TextDecoder,ho=(c,f,S,T)=>{var U=f+S;if(T)return U;for(;c[f]&&!(f>=U);)++f;return f},hc=(c,f=0,S,T)=>{f>>>=0;var U=ho(c,f,S,T);if(U-f>16&&c.buffer&&fo)return fo.decode(c.subarray(f,U));for(var k="";f<U;){var H=c[f++];if(!(H&128)){k+=String.fromCharCode(H);continue}var X=c[f++]&63;if((H&224)==192){k+=String.fromCharCode((H&31)<<6|X);continue}var J=c[f++]&63;if((H&240)==224?H=(H&15)<<12|X<<6|J:H=(H&7)<<18|X<<12|J<<6|c[f++]&63,H<65536)k+=String.fromCharCode(H);else{var he=H-65536;k+=String.fromCharCode(55296|he>>10,56320|he&1023)}}return k},dc=(c,f,S)=>(c>>>=0,c?hc(P,c,f,S):"");function pc(c,f){c>>>=0,f>>>=0,f=Me(f),ce(c,{name:f,fromWireType(S){var T=x[S>>>2>>>0],U=S+4,k;return k=dc(U,T,!0),Zn(S),k},toWireType(S,T){T instanceof ArrayBuffer&&(T=new Uint8Array(T));var U,k=typeof T=="string";k||ArrayBuffer.isView(T)&&T.BYTES_PER_ELEMENT==1||me("Cannot pass non-string to std::string"),k?U=fc(T):U=T.length;var H=xs(4+U+1),X=H+4;return x[H>>>2>>>0]=U,k?uc(T,X,U+1):P.set(T,X>>>0),S!==null&&S.push(Zn,H),H},readValueFromPointer:gt,destructorFunction(S){Zn(S)}})}var po=globalThis.TextDecoder?new TextDecoder("utf-16le"):void 0,mc=(c,f,S)=>{var T=c>>>1,U=ho(I,T,f/2,S);if(U-T>16&&po)return po.decode(I.subarray(T>>>0,U>>>0));for(var k="",H=T;H<U;++H){var X=I[H>>>0];k+=String.fromCharCode(X)}return k},_c=(c,f,S)=>{if(S??=2147483647,S<2)return 0;S-=2;for(var T=f,U=S<c.length*2?S/2:c.length,k=0;k<U;++k){var H=c.charCodeAt(k);D[f>>>1>>>0]=H,f+=2}return D[f>>>1>>>0]=0,f-T},gc=c=>c.length*2,vc=(c,f,S)=>{for(var T="",U=c>>>2,k=0;!(k>=f/4);k++){var H=x[U+k>>>0];if(!H&&!S)break;T+=String.fromCodePoint(H)}return T},xc=(c,f,S)=>{if(f>>>=0,S??=2147483647,S<4)return 0;for(var T=f,U=T+S-4,k=0;k<c.length;++k){var H=c.codePointAt(k);if(H>65535&&k++,G[f>>>2>>>0]=H,f+=4,f+4>U)break}return G[f>>>2>>>0]=0,f-T},Mc=c=>{for(var f=0,S=0;S<c.length;++S){var T=c.codePointAt(S);T>65535&&S++,f+=4}return f};function Sc(c,f,S){c>>>=0,f>>>=0,S>>>=0,S=Me(S);var T,U,k;f===2?(T=mc,U=_c,k=gc):(T=vc,U=xc,k=Mc),ce(c,{name:S,fromWireType:H=>{var X=x[H>>>2>>>0],J=T(H+4,X*f,!0);return Zn(H),J},toWireType:(H,X)=>{typeof X!="string"&&me(`Cannot pass non-string to C++ string type ${S}`);var J=k(X),he=xs(4+J+f);return x[he>>>2>>>0]=J/f,U(X,he+4,J+f),H!==null&&H.push(Zn,he),he},readValueFromPointer:gt,destructorFunction(H){Zn(H)}})}function yc(c,f,S,T,U,k){c>>>=0,f>>>=0,S>>>=0,T>>>=0,U>>>=0,k>>>=0,It[c]={name:Me(f),rawConstructor:cn(S,T),rawDestructor:cn(U,k),fields:[]}}function Ec(c,f,S,T,U,k,H,X,J,he){c>>>=0,f>>>=0,S>>>=0,T>>>=0,U>>>=0,k>>>=0,H>>>=0,X>>>=0,J>>>=0,he>>>=0,It[c].fields.push({fieldName:Me(f),getterReturnType:S,getter:cn(T,U),getterContext:k,setterArgumentType:H,setter:cn(X,J),setterContext:he})}var bc=function(c,f){c>>>=0,f>>>=0,f=Me(f),ce(c,{isVoid:!0,name:f,fromWireType:()=>{},toWireType:(S,T)=>{}})};function Tc(c,f){c>>>=0,f>>>=0,c=je.toValue(c),f=je.toValue(f),c.set(f)}var _s=[],Ac=c=>{var f=_s.length;return _s.push(c),f},wc=(c,f)=>{for(var S=new Array(c),T=0;T<c;++T)S[T]=qt(x[f+T*4>>>2>>>0],`parameter ${T}`);return S},Rc=(c,f,S)=>{var T=[],U=c(T,S);return T.length&&(x[f>>>2>>>0]=je.toHandle(T)),U},Cc={},mo=c=>{var f=Cc[c];return f===void 0?Me(c):f},Pc=function(c,f,S){f>>>=0;var T=8,[U,...k]=wc(c,f),H=U.toWireType.bind(U),X=k.map(ge=>ge.readValueFromPointer.bind(ge));c--;var J=new Array(c),he=(ge,Ce,Ue,ze)=>{for(var Xe=0,qe=0;qe<c;++qe)J[qe]=X[qe](ze+Xe),Xe+=T;var lt;switch(S){case 0:lt=je.toValue(ge).apply(null,J);break;case 2:lt=Reflect.construct(je.toValue(ge),J);break;case 3:lt=J[0];break;case 1:lt=je.toValue(ge)[mo(Ce)](...J);break}return Rc(H,Ue,lt)},pe=`methodCaller<(${k.map(ge=>ge.name)}) => ${U.name}>`;return Ac(Ke(pe,he))};function Dc(c,f){return c>>>=0,f>>>=0,c=je.toValue(c),f=je.toValue(f),c==f}function Lc(c,f){return c>>>=0,f>>>=0,c=je.toValue(c),f=je.toValue(f),je.toHandle(c[f])}function Uc(c){c>>>=0,c>9&&(Et[c+1]+=1)}function Ic(c,f,S,T,U){return c>>>=0,f>>>=0,S>>>=0,T>>>=0,U>>>=0,_s[c](f,S,T,U)}function Fc(c){return c>>>=0,je.toHandle(mo(c))}function Nc(){return je.toHandle({})}function Oc(c){c>>>=0;var f=je.toValue(c);mt(f),Lt(c)}function Bc(c,f,S){c>>>=0,f>>>=0,S>>>=0,c=je.toValue(c),f=je.toValue(f),S=je.toValue(S),c[f]=S}var Vc=()=>4294901760,zc=(c,f)=>Math.ceil(c/f)*f,Gc=c=>{var f=wr.buffer.byteLength,S=(c-f+65535)/65536|0;try{return wr.grow(S),oe(),1}catch{}};function Hc(c){c>>>=0;var f=P.length,S=Vc();if(c>S)return!1;for(var T=1;T<=4;T*=2){var U=f*(1+.2/T);U=Math.min(U,c+100663296);var k=Math.min(S,zc(Math.max(c,U),65536)),H=Gc(k);if(H)return!0}return!1}var kc=(c,f)=>{if(li)for(var S=c;S<c+f;S++){var T=Ti(S);T&&li.set(T,S)}},li,Wc=c=>(li||(li=new WeakMap,kc(0,nr.length)),li.get(c)||0),gs=[],Xc=()=>gs.length?gs.pop():nr.grow(1),vs=(c,f)=>nr.set(c,f),_o=c=>{const f=c.length;return[f%128|128,f>>7,...c]},$c={i:127,p:127,j:126,f:125,d:124,e:111},go=c=>_o(Array.from(c,f=>{var S=$c[f];return S})),qc=(c,f)=>{var S=Uint8Array.of(0,97,115,109,1,0,0,0,1,..._o([1,96,...go(f.slice(1)),...go(f[0]==="v"?"":f[0])]),2,7,1,1,101,1,102,0,0,7,5,1,1,102,0,0),T=new WebAssembly.Module(S),U=new WebAssembly.Instance(T,{e:{f:c}}),k=U.exports.f;return k},vo=(c,f)=>{var S=Wc(c);if(S)return S;var T=Xc();try{vs(T,c)}catch(k){if(!(k instanceof TypeError))throw k;var U=qc(c,f);vs(T,U)}return li.set(c,T),T},xo=c=>{li.delete(Ti(c)),vs(c,null),gs.push(c)};if(le(),Tr(),t.noExitRuntime&&t.noExitRuntime,t.print&&t.print,t.printErr&&(y=t.printErr),t.wasmBinary&&(b=t.wasmBinary),t.arguments&&t.arguments,t.thisProgram&&t.thisProgram,t.preInit)for(typeof t.preInit=="function"&&(t.preInit=[t.preInit]);t.preInit.length>0;)t.preInit.shift()();t.addFunction=vo,t.removeFunction=xo;var Mo,xs,Zn,wr,nr;function Yc(c){Mo=c.L,xs=c.N,Zn=c.O,wr=c.J,nr=c.M}var Kc={r:Ft,C:F,n:ee,x:Be,F:Ge,h:q,i:He,a:We,I:ot,D:St,v:jn,p:oi,w:Yt,c:sn,m:Wt,k:tr,d:Bn,l:lc,E:pc,t:Sc,o:yc,j:Ec,G:bc,s:Tc,g:Pc,b:Lt,q:Dc,H:Lc,z:Uc,f:Ic,u:Fc,A:Nc,e:Oc,y:Bc,B:Hc};function jc(c){c=Object.assign({},c);var f=T=>U=>T(U)>>>0,S=T=>()=>T()>>>0;return c.L=f(c.L),c.N=f(c.N),c._emscripten_stack_alloc=f(c._emscripten_stack_alloc),c.emscripten_stack_get_current=S(c.emscripten_stack_get_current),c}function Zc(){Q();function c(){t.calledRun=!0,!g&&(te(),R?.(t),t.onRuntimeInitialized?.(),ue())}t.setStatus?(t.setStatus("Running..."),setTimeout(()=>{setTimeout(()=>t.setStatus(""),1),c()},1)):c()}var ci;return ci=await fe(),Zc(),ie?e=t:e=new Promise((c,f)=>{R=c,C=f}),e}const qa="182",ru=0,yo=1,su=2,jr=1,au=2,ur=3,ai=0,on=1,Wn=2,$n=0,zi=1,Eo=2,bo=3,To=4,ou=5,gi=100,lu=101,cu=102,uu=103,fu=104,hu=200,du=201,pu=202,mu=203,Js=204,Qs=205,_u=206,gu=207,vu=208,xu=209,Mu=210,Su=211,yu=212,Eu=213,bu=214,ea=0,ta=1,na=2,Hi=3,ia=4,ra=5,sa=6,aa=7,Ya=0,Tu=1,Au=2,Pn=0,yl=1,El=2,bl=3,Tl=4,Al=5,wl=6,Rl=7,Cl=300,Si=301,ki=302,oa=303,la=304,ss=306,ca=1e3,Xn=1001,ua=1002,Xt=1003,wu=1004,Cr=1005,Qt=1006,Ms=1007,xi=1008,dn=1009,Pl=1010,Dl=1011,hr=1012,Ka=1013,Ln=1014,Rn=1015,Yn=1016,ja=1017,Za=1018,dr=1020,Ll=35902,Ul=35899,Il=1021,Fl=1022,En=1023,Kn=1026,Mi=1027,Nl=1028,Ja=1029,Wi=1030,Qa=1031,eo=1033,Zr=33776,Jr=33777,Qr=33778,es=33779,fa=35840,ha=35841,da=35842,pa=35843,ma=36196,_a=37492,ga=37496,va=37488,xa=37489,Ma=37490,Sa=37491,ya=37808,Ea=37809,ba=37810,Ta=37811,Aa=37812,wa=37813,Ra=37814,Ca=37815,Pa=37816,Da=37817,La=37818,Ua=37819,Ia=37820,Fa=37821,Na=36492,Oa=36494,Ba=36495,Va=36283,za=36284,Ga=36285,Ha=36286,Ru=3200,to=0,Cu=1,ri="",mn="srgb",Xi="srgb-linear",ns="linear",wt="srgb",Ai=7680,Ao=519,Pu=512,Du=513,Lu=514,no=515,Uu=516,Iu=517,io=518,Fu=519,wo=35044,Ro="300 es",Cn=2e3,is=2001;function Ol(i){for(let e=i.length-1;e>=0;--e)if(i[e]>=65535)return!0;return!1}function rs(i){return document.createElementNS("http://www.w3.org/1999/xhtml",i)}function Nu(){const i=rs("canvas");return i.style.display="block",i}const Co={};function Po(...i){const e="THREE."+i.shift();console.log(e,...i)}function et(...i){const e="THREE."+i.shift();console.warn(e,...i)}function xt(...i){const e="THREE."+i.shift();console.error(e,...i)}function pr(...i){const e=i.join(" ");e in Co||(Co[e]=!0,et(...i))}function Ou(i,e,t){return new Promise(function(n,r){function s(){switch(i.clientWaitSync(e,i.SYNC_FLUSH_COMMANDS_BIT,0)){case i.WAIT_FAILED:r();break;case i.TIMEOUT_EXPIRED:setTimeout(s,t);break;default:n()}}setTimeout(s,t)})}class qi{addEventListener(e,t){this._listeners===void 0&&(this._listeners={});const n=this._listeners;n[e]===void 0&&(n[e]=[]),n[e].indexOf(t)===-1&&n[e].push(t)}hasEventListener(e,t){const n=this._listeners;return n===void 0?!1:n[e]!==void 0&&n[e].indexOf(t)!==-1}removeEventListener(e,t){const n=this._listeners;if(n===void 0)return;const r=n[e];if(r!==void 0){const s=r.indexOf(t);s!==-1&&r.splice(s,1)}}dispatchEvent(e){const t=this._listeners;if(t===void 0)return;const n=t[e.type];if(n!==void 0){e.target=this;const r=n.slice(0);for(let s=0,a=r.length;s<a;s++)r[s].call(this,e);e.target=null}}}const Zt=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"],Ss=Math.PI/180,ka=180/Math.PI;function gr(){const i=Math.random()*4294967295|0,e=Math.random()*4294967295|0,t=Math.random()*4294967295|0,n=Math.random()*4294967295|0;return(Zt[i&255]+Zt[i>>8&255]+Zt[i>>16&255]+Zt[i>>24&255]+"-"+Zt[e&255]+Zt[e>>8&255]+"-"+Zt[e>>16&15|64]+Zt[e>>24&255]+"-"+Zt[t&63|128]+Zt[t>>8&255]+"-"+Zt[t>>16&255]+Zt[t>>24&255]+Zt[n&255]+Zt[n>>8&255]+Zt[n>>16&255]+Zt[n>>24&255]).toLowerCase()}function dt(i,e,t){return Math.max(e,Math.min(t,i))}function Bu(i,e){return(i%e+e)%e}function ys(i,e,t){return(1-t)*i+t*e}function ir(i,e){switch(e.constructor){case Float32Array:return i;case Uint32Array:return i/4294967295;case Uint16Array:return i/65535;case Uint8Array:return i/255;case Int32Array:return Math.max(i/2147483647,-1);case Int16Array:return Math.max(i/32767,-1);case Int8Array:return Math.max(i/127,-1);default:throw new Error("Invalid component type.")}}function an(i,e){switch(e.constructor){case Float32Array:return i;case Uint32Array:return Math.round(i*4294967295);case Uint16Array:return Math.round(i*65535);case Uint8Array:return Math.round(i*255);case Int32Array:return Math.round(i*2147483647);case Int16Array:return Math.round(i*32767);case Int8Array:return Math.round(i*127);default:throw new Error("Invalid component type.")}}class pt{constructor(e=0,t=0){pt.prototype.isVector2=!0,this.x=e,this.y=t}get width(){return this.x}set width(e){this.x=e}get height(){return this.y}set height(e){this.y=e}set(e,t){return this.x=e,this.y=t,this}setScalar(e){return this.x=e,this.y=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y)}copy(e){return this.x=e.x,this.y=e.y,this}add(e){return this.x+=e.x,this.y+=e.y,this}addScalar(e){return this.x+=e,this.y+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this}subScalar(e){return this.x-=e,this.y-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this}multiply(e){return this.x*=e.x,this.y*=e.y,this}multiplyScalar(e){return this.x*=e,this.y*=e,this}divide(e){return this.x/=e.x,this.y/=e.y,this}divideScalar(e){return this.multiplyScalar(1/e)}applyMatrix3(e){const t=this.x,n=this.y,r=e.elements;return this.x=r[0]*t+r[3]*n+r[6],this.y=r[1]*t+r[4]*n+r[7],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this}clamp(e,t){return this.x=dt(this.x,e.x,t.x),this.y=dt(this.y,e.y,t.y),this}clampScalar(e,t){return this.x=dt(this.x,e,t),this.y=dt(this.y,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(dt(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(e){return this.x*e.x+this.y*e.y}cross(e){return this.x*e.y-this.y*e.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(e){const t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;const n=this.dot(e)/t;return Math.acos(dt(n,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const t=this.x-e.x,n=this.y-e.y;return t*t+n*n}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this}equals(e){return e.x===this.x&&e.y===this.y}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this}rotateAround(e,t){const n=Math.cos(t),r=Math.sin(t),s=this.x-e.x,a=this.y-e.y;return this.x=s*n-a*r+e.x,this.y=s*r+a*n+e.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}}class vr{constructor(e=0,t=0,n=0,r=1){this.isQuaternion=!0,this._x=e,this._y=t,this._z=n,this._w=r}static slerpFlat(e,t,n,r,s,a,o){let u=n[r+0],l=n[r+1],h=n[r+2],p=n[r+3],m=s[a+0],v=s[a+1],y=s[a+2],b=s[a+3];if(o<=0){e[t+0]=u,e[t+1]=l,e[t+2]=h,e[t+3]=p;return}if(o>=1){e[t+0]=m,e[t+1]=v,e[t+2]=y,e[t+3]=b;return}if(p!==b||u!==m||l!==v||h!==y){let g=u*m+l*v+h*y+p*b;g<0&&(m=-m,v=-v,y=-y,b=-b,g=-g);let d=1-o;if(g<.9995){const R=Math.acos(g),C=Math.sin(R);d=Math.sin(d*R)/C,o=Math.sin(o*R)/C,u=u*d+m*o,l=l*d+v*o,h=h*d+y*o,p=p*d+b*o}else{u=u*d+m*o,l=l*d+v*o,h=h*d+y*o,p=p*d+b*o;const R=1/Math.sqrt(u*u+l*l+h*h+p*p);u*=R,l*=R,h*=R,p*=R}}e[t]=u,e[t+1]=l,e[t+2]=h,e[t+3]=p}static multiplyQuaternionsFlat(e,t,n,r,s,a){const o=n[r],u=n[r+1],l=n[r+2],h=n[r+3],p=s[a],m=s[a+1],v=s[a+2],y=s[a+3];return e[t]=o*y+h*p+u*v-l*m,e[t+1]=u*y+h*m+l*p-o*v,e[t+2]=l*y+h*v+o*m-u*p,e[t+3]=h*y-o*p-u*m-l*v,e}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get w(){return this._w}set w(e){this._w=e,this._onChangeCallback()}set(e,t,n,r){return this._x=e,this._y=t,this._z=n,this._w=r,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(e){return this._x=e.x,this._y=e.y,this._z=e.z,this._w=e.w,this._onChangeCallback(),this}setFromEuler(e,t=!0){const n=e._x,r=e._y,s=e._z,a=e._order,o=Math.cos,u=Math.sin,l=o(n/2),h=o(r/2),p=o(s/2),m=u(n/2),v=u(r/2),y=u(s/2);switch(a){case"XYZ":this._x=m*h*p+l*v*y,this._y=l*v*p-m*h*y,this._z=l*h*y+m*v*p,this._w=l*h*p-m*v*y;break;case"YXZ":this._x=m*h*p+l*v*y,this._y=l*v*p-m*h*y,this._z=l*h*y-m*v*p,this._w=l*h*p+m*v*y;break;case"ZXY":this._x=m*h*p-l*v*y,this._y=l*v*p+m*h*y,this._z=l*h*y+m*v*p,this._w=l*h*p-m*v*y;break;case"ZYX":this._x=m*h*p-l*v*y,this._y=l*v*p+m*h*y,this._z=l*h*y-m*v*p,this._w=l*h*p+m*v*y;break;case"YZX":this._x=m*h*p+l*v*y,this._y=l*v*p+m*h*y,this._z=l*h*y-m*v*p,this._w=l*h*p-m*v*y;break;case"XZY":this._x=m*h*p-l*v*y,this._y=l*v*p-m*h*y,this._z=l*h*y+m*v*p,this._w=l*h*p+m*v*y;break;default:et("Quaternion: .setFromEuler() encountered an unknown order: "+a)}return t===!0&&this._onChangeCallback(),this}setFromAxisAngle(e,t){const n=t/2,r=Math.sin(n);return this._x=e.x*r,this._y=e.y*r,this._z=e.z*r,this._w=Math.cos(n),this._onChangeCallback(),this}setFromRotationMatrix(e){const t=e.elements,n=t[0],r=t[4],s=t[8],a=t[1],o=t[5],u=t[9],l=t[2],h=t[6],p=t[10],m=n+o+p;if(m>0){const v=.5/Math.sqrt(m+1);this._w=.25/v,this._x=(h-u)*v,this._y=(s-l)*v,this._z=(a-r)*v}else if(n>o&&n>p){const v=2*Math.sqrt(1+n-o-p);this._w=(h-u)/v,this._x=.25*v,this._y=(r+a)/v,this._z=(s+l)/v}else if(o>p){const v=2*Math.sqrt(1+o-n-p);this._w=(s-l)/v,this._x=(r+a)/v,this._y=.25*v,this._z=(u+h)/v}else{const v=2*Math.sqrt(1+p-n-o);this._w=(a-r)/v,this._x=(s+l)/v,this._y=(u+h)/v,this._z=.25*v}return this._onChangeCallback(),this}setFromUnitVectors(e,t){let n=e.dot(t)+1;return n<1e-8?(n=0,Math.abs(e.x)>Math.abs(e.z)?(this._x=-e.y,this._y=e.x,this._z=0,this._w=n):(this._x=0,this._y=-e.z,this._z=e.y,this._w=n)):(this._x=e.y*t.z-e.z*t.y,this._y=e.z*t.x-e.x*t.z,this._z=e.x*t.y-e.y*t.x,this._w=n),this.normalize()}angleTo(e){return 2*Math.acos(Math.abs(dt(this.dot(e),-1,1)))}rotateTowards(e,t){const n=this.angleTo(e);if(n===0)return this;const r=Math.min(1,t/n);return this.slerp(e,r),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(e){return this._x*e._x+this._y*e._y+this._z*e._z+this._w*e._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let e=this.length();return e===0?(this._x=0,this._y=0,this._z=0,this._w=1):(e=1/e,this._x=this._x*e,this._y=this._y*e,this._z=this._z*e,this._w=this._w*e),this._onChangeCallback(),this}multiply(e){return this.multiplyQuaternions(this,e)}premultiply(e){return this.multiplyQuaternions(e,this)}multiplyQuaternions(e,t){const n=e._x,r=e._y,s=e._z,a=e._w,o=t._x,u=t._y,l=t._z,h=t._w;return this._x=n*h+a*o+r*l-s*u,this._y=r*h+a*u+s*o-n*l,this._z=s*h+a*l+n*u-r*o,this._w=a*h-n*o-r*u-s*l,this._onChangeCallback(),this}slerp(e,t){if(t<=0)return this;if(t>=1)return this.copy(e);let n=e._x,r=e._y,s=e._z,a=e._w,o=this.dot(e);o<0&&(n=-n,r=-r,s=-s,a=-a,o=-o);let u=1-t;if(o<.9995){const l=Math.acos(o),h=Math.sin(l);u=Math.sin(u*l)/h,t=Math.sin(t*l)/h,this._x=this._x*u+n*t,this._y=this._y*u+r*t,this._z=this._z*u+s*t,this._w=this._w*u+a*t,this._onChangeCallback()}else this._x=this._x*u+n*t,this._y=this._y*u+r*t,this._z=this._z*u+s*t,this._w=this._w*u+a*t,this.normalize();return this}slerpQuaternions(e,t,n){return this.copy(e).slerp(t,n)}random(){const e=2*Math.PI*Math.random(),t=2*Math.PI*Math.random(),n=Math.random(),r=Math.sqrt(1-n),s=Math.sqrt(n);return this.set(r*Math.sin(e),r*Math.cos(e),s*Math.sin(t),s*Math.cos(t))}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._w===this._w}fromArray(e,t=0){return this._x=e[t],this._y=e[t+1],this._z=e[t+2],this._w=e[t+3],this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._w,e}fromBufferAttribute(e,t){return this._x=e.getX(t),this._y=e.getY(t),this._z=e.getZ(t),this._w=e.getW(t),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}}class ${constructor(e=0,t=0,n=0){$.prototype.isVector3=!0,this.x=e,this.y=t,this.z=n}set(e,t,n){return n===void 0&&(n=this.z),this.x=e,this.y=t,this.z=n,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this}multiplyVectors(e,t){return this.x=e.x*t.x,this.y=e.y*t.y,this.z=e.z*t.z,this}applyEuler(e){return this.applyQuaternion(Do.setFromEuler(e))}applyAxisAngle(e,t){return this.applyQuaternion(Do.setFromAxisAngle(e,t))}applyMatrix3(e){const t=this.x,n=this.y,r=this.z,s=e.elements;return this.x=s[0]*t+s[3]*n+s[6]*r,this.y=s[1]*t+s[4]*n+s[7]*r,this.z=s[2]*t+s[5]*n+s[8]*r,this}applyNormalMatrix(e){return this.applyMatrix3(e).normalize()}applyMatrix4(e){const t=this.x,n=this.y,r=this.z,s=e.elements,a=1/(s[3]*t+s[7]*n+s[11]*r+s[15]);return this.x=(s[0]*t+s[4]*n+s[8]*r+s[12])*a,this.y=(s[1]*t+s[5]*n+s[9]*r+s[13])*a,this.z=(s[2]*t+s[6]*n+s[10]*r+s[14])*a,this}applyQuaternion(e){const t=this.x,n=this.y,r=this.z,s=e.x,a=e.y,o=e.z,u=e.w,l=2*(a*r-o*n),h=2*(o*t-s*r),p=2*(s*n-a*t);return this.x=t+u*l+a*p-o*h,this.y=n+u*h+o*l-s*p,this.z=r+u*p+s*h-a*l,this}project(e){return this.applyMatrix4(e.matrixWorldInverse).applyMatrix4(e.projectionMatrix)}unproject(e){return this.applyMatrix4(e.projectionMatrixInverse).applyMatrix4(e.matrixWorld)}transformDirection(e){const t=this.x,n=this.y,r=this.z,s=e.elements;return this.x=s[0]*t+s[4]*n+s[8]*r,this.y=s[1]*t+s[5]*n+s[9]*r,this.z=s[2]*t+s[6]*n+s[10]*r,this.normalize()}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this}divideScalar(e){return this.multiplyScalar(1/e)}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this}clamp(e,t){return this.x=dt(this.x,e.x,t.x),this.y=dt(this.y,e.y,t.y),this.z=dt(this.z,e.z,t.z),this}clampScalar(e,t){return this.x=dt(this.x,e,t),this.y=dt(this.y,e,t),this.z=dt(this.z,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(dt(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this.z=e.z+(t.z-e.z)*n,this}cross(e){return this.crossVectors(this,e)}crossVectors(e,t){const n=e.x,r=e.y,s=e.z,a=t.x,o=t.y,u=t.z;return this.x=r*u-s*o,this.y=s*a-n*u,this.z=n*o-r*a,this}projectOnVector(e){const t=e.lengthSq();if(t===0)return this.set(0,0,0);const n=e.dot(this)/t;return this.copy(e).multiplyScalar(n)}projectOnPlane(e){return Es.copy(this).projectOnVector(e),this.sub(Es)}reflect(e){return this.sub(Es.copy(e).multiplyScalar(2*this.dot(e)))}angleTo(e){const t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;const n=this.dot(e)/t;return Math.acos(dt(n,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const t=this.x-e.x,n=this.y-e.y,r=this.z-e.z;return t*t+n*n+r*r}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)+Math.abs(this.z-e.z)}setFromSpherical(e){return this.setFromSphericalCoords(e.radius,e.phi,e.theta)}setFromSphericalCoords(e,t,n){const r=Math.sin(t)*e;return this.x=r*Math.sin(n),this.y=Math.cos(t)*e,this.z=r*Math.cos(n),this}setFromCylindrical(e){return this.setFromCylindricalCoords(e.radius,e.theta,e.y)}setFromCylindricalCoords(e,t,n){return this.x=e*Math.sin(t),this.y=n,this.z=e*Math.cos(t),this}setFromMatrixPosition(e){const t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this}setFromMatrixScale(e){const t=this.setFromMatrixColumn(e,0).length(),n=this.setFromMatrixColumn(e,1).length(),r=this.setFromMatrixColumn(e,2).length();return this.x=t,this.y=n,this.z=r,this}setFromMatrixColumn(e,t){return this.fromArray(e.elements,t*4)}setFromMatrix3Column(e,t){return this.fromArray(e.elements,t*3)}setFromEuler(e){return this.x=e._x,this.y=e._y,this.z=e._z,this}setFromColor(e){return this.x=e.r,this.y=e.g,this.z=e.b,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){const e=Math.random()*Math.PI*2,t=Math.random()*2-1,n=Math.sqrt(1-t*t);return this.x=n*Math.cos(e),this.y=t,this.z=n*Math.sin(e),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}}const Es=new $,Do=new vr;class st{constructor(e,t,n,r,s,a,o,u,l){st.prototype.isMatrix3=!0,this.elements=[1,0,0,0,1,0,0,0,1],e!==void 0&&this.set(e,t,n,r,s,a,o,u,l)}set(e,t,n,r,s,a,o,u,l){const h=this.elements;return h[0]=e,h[1]=r,h[2]=o,h[3]=t,h[4]=s,h[5]=u,h[6]=n,h[7]=a,h[8]=l,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(e){const t=this.elements,n=e.elements;return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=n[3],t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=n[7],t[8]=n[8],this}extractBasis(e,t,n){return e.setFromMatrix3Column(this,0),t.setFromMatrix3Column(this,1),n.setFromMatrix3Column(this,2),this}setFromMatrix4(e){const t=e.elements;return this.set(t[0],t[4],t[8],t[1],t[5],t[9],t[2],t[6],t[10]),this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){const n=e.elements,r=t.elements,s=this.elements,a=n[0],o=n[3],u=n[6],l=n[1],h=n[4],p=n[7],m=n[2],v=n[5],y=n[8],b=r[0],g=r[3],d=r[6],R=r[1],C=r[4],A=r[7],P=r[2],D=r[5],I=r[8];return s[0]=a*b+o*R+u*P,s[3]=a*g+o*C+u*D,s[6]=a*d+o*A+u*I,s[1]=l*b+h*R+p*P,s[4]=l*g+h*C+p*D,s[7]=l*d+h*A+p*I,s[2]=m*b+v*R+y*P,s[5]=m*g+v*C+y*D,s[8]=m*d+v*A+y*I,this}multiplyScalar(e){const t=this.elements;return t[0]*=e,t[3]*=e,t[6]*=e,t[1]*=e,t[4]*=e,t[7]*=e,t[2]*=e,t[5]*=e,t[8]*=e,this}determinant(){const e=this.elements,t=e[0],n=e[1],r=e[2],s=e[3],a=e[4],o=e[5],u=e[6],l=e[7],h=e[8];return t*a*h-t*o*l-n*s*h+n*o*u+r*s*l-r*a*u}invert(){const e=this.elements,t=e[0],n=e[1],r=e[2],s=e[3],a=e[4],o=e[5],u=e[6],l=e[7],h=e[8],p=h*a-o*l,m=o*u-h*s,v=l*s-a*u,y=t*p+n*m+r*v;if(y===0)return this.set(0,0,0,0,0,0,0,0,0);const b=1/y;return e[0]=p*b,e[1]=(r*l-h*n)*b,e[2]=(o*n-r*a)*b,e[3]=m*b,e[4]=(h*t-r*u)*b,e[5]=(r*s-o*t)*b,e[6]=v*b,e[7]=(n*u-l*t)*b,e[8]=(a*t-n*s)*b,this}transpose(){let e;const t=this.elements;return e=t[1],t[1]=t[3],t[3]=e,e=t[2],t[2]=t[6],t[6]=e,e=t[5],t[5]=t[7],t[7]=e,this}getNormalMatrix(e){return this.setFromMatrix4(e).invert().transpose()}transposeIntoArray(e){const t=this.elements;return e[0]=t[0],e[1]=t[3],e[2]=t[6],e[3]=t[1],e[4]=t[4],e[5]=t[7],e[6]=t[2],e[7]=t[5],e[8]=t[8],this}setUvTransform(e,t,n,r,s,a,o){const u=Math.cos(s),l=Math.sin(s);return this.set(n*u,n*l,-n*(u*a+l*o)+a+e,-r*l,r*u,-r*(-l*a+u*o)+o+t,0,0,1),this}scale(e,t){return this.premultiply(bs.makeScale(e,t)),this}rotate(e){return this.premultiply(bs.makeRotation(-e)),this}translate(e,t){return this.premultiply(bs.makeTranslation(e,t)),this}makeTranslation(e,t){return e.isVector2?this.set(1,0,e.x,0,1,e.y,0,0,1):this.set(1,0,e,0,1,t,0,0,1),this}makeRotation(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,-n,0,n,t,0,0,0,1),this}makeScale(e,t){return this.set(e,0,0,0,t,0,0,0,1),this}equals(e){const t=this.elements,n=e.elements;for(let r=0;r<9;r++)if(t[r]!==n[r])return!1;return!0}fromArray(e,t=0){for(let n=0;n<9;n++)this.elements[n]=e[n+t];return this}toArray(e=[],t=0){const n=this.elements;return e[t]=n[0],e[t+1]=n[1],e[t+2]=n[2],e[t+3]=n[3],e[t+4]=n[4],e[t+5]=n[5],e[t+6]=n[6],e[t+7]=n[7],e[t+8]=n[8],e}clone(){return new this.constructor().fromArray(this.elements)}}const bs=new st,Lo=new st().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),Uo=new st().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);function Vu(){const i={enabled:!0,workingColorSpace:Xi,spaces:{},convert:function(r,s,a){return this.enabled===!1||s===a||!s||!a||(this.spaces[s].transfer===wt&&(r.r=qn(r.r),r.g=qn(r.g),r.b=qn(r.b)),this.spaces[s].primaries!==this.spaces[a].primaries&&(r.applyMatrix3(this.spaces[s].toXYZ),r.applyMatrix3(this.spaces[a].fromXYZ)),this.spaces[a].transfer===wt&&(r.r=Gi(r.r),r.g=Gi(r.g),r.b=Gi(r.b))),r},workingToColorSpace:function(r,s){return this.convert(r,this.workingColorSpace,s)},colorSpaceToWorking:function(r,s){return this.convert(r,s,this.workingColorSpace)},getPrimaries:function(r){return this.spaces[r].primaries},getTransfer:function(r){return r===ri?ns:this.spaces[r].transfer},getToneMappingMode:function(r){return this.spaces[r].outputColorSpaceConfig.toneMappingMode||"standard"},getLuminanceCoefficients:function(r,s=this.workingColorSpace){return r.fromArray(this.spaces[s].luminanceCoefficients)},define:function(r){Object.assign(this.spaces,r)},_getMatrix:function(r,s,a){return r.copy(this.spaces[s].toXYZ).multiply(this.spaces[a].fromXYZ)},_getDrawingBufferColorSpace:function(r){return this.spaces[r].outputColorSpaceConfig.drawingBufferColorSpace},_getUnpackColorSpace:function(r=this.workingColorSpace){return this.spaces[r].workingColorSpaceConfig.unpackColorSpace},fromWorkingColorSpace:function(r,s){return pr("ColorManagement: .fromWorkingColorSpace() has been renamed to .workingToColorSpace()."),i.workingToColorSpace(r,s)},toWorkingColorSpace:function(r,s){return pr("ColorManagement: .toWorkingColorSpace() has been renamed to .colorSpaceToWorking()."),i.colorSpaceToWorking(r,s)}},e=[.64,.33,.3,.6,.15,.06],t=[.2126,.7152,.0722],n=[.3127,.329];return i.define({[Xi]:{primaries:e,whitePoint:n,transfer:ns,toXYZ:Lo,fromXYZ:Uo,luminanceCoefficients:t,workingColorSpaceConfig:{unpackColorSpace:mn},outputColorSpaceConfig:{drawingBufferColorSpace:mn}},[mn]:{primaries:e,whitePoint:n,transfer:wt,toXYZ:Lo,fromXYZ:Uo,luminanceCoefficients:t,outputColorSpaceConfig:{drawingBufferColorSpace:mn}}}),i}const _t=Vu();function qn(i){return i<.04045?i*.0773993808:Math.pow(i*.9478672986+.0521327014,2.4)}function Gi(i){return i<.0031308?i*12.92:1.055*Math.pow(i,.41666)-.055}let wi;class zu{static getDataURL(e,t="image/png"){if(/^data:/i.test(e.src)||typeof HTMLCanvasElement>"u")return e.src;let n;if(e instanceof HTMLCanvasElement)n=e;else{wi===void 0&&(wi=rs("canvas")),wi.width=e.width,wi.height=e.height;const r=wi.getContext("2d");e instanceof ImageData?r.putImageData(e,0,0):r.drawImage(e,0,0,e.width,e.height),n=wi}return n.toDataURL(t)}static sRGBToLinear(e){if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&e instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&e instanceof ImageBitmap){const t=rs("canvas");t.width=e.width,t.height=e.height;const n=t.getContext("2d");n.drawImage(e,0,0,e.width,e.height);const r=n.getImageData(0,0,e.width,e.height),s=r.data;for(let a=0;a<s.length;a++)s[a]=qn(s[a]/255)*255;return n.putImageData(r,0,0),t}else if(e.data){const t=e.data.slice(0);for(let n=0;n<t.length;n++)t instanceof Uint8Array||t instanceof Uint8ClampedArray?t[n]=Math.floor(qn(t[n]/255)*255):t[n]=qn(t[n]);return{data:t,width:e.width,height:e.height}}else return et("ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),e}}let Gu=0;class ro{constructor(e=null){this.isSource=!0,Object.defineProperty(this,"id",{value:Gu++}),this.uuid=gr(),this.data=e,this.dataReady=!0,this.version=0}getSize(e){const t=this.data;return typeof HTMLVideoElement<"u"&&t instanceof HTMLVideoElement?e.set(t.videoWidth,t.videoHeight,0):typeof VideoFrame<"u"&&t instanceof VideoFrame?e.set(t.displayHeight,t.displayWidth,0):t!==null?e.set(t.width,t.height,t.depth||0):e.set(0,0,0),e}set needsUpdate(e){e===!0&&this.version++}toJSON(e){const t=e===void 0||typeof e=="string";if(!t&&e.images[this.uuid]!==void 0)return e.images[this.uuid];const n={uuid:this.uuid,url:""},r=this.data;if(r!==null){let s;if(Array.isArray(r)){s=[];for(let a=0,o=r.length;a<o;a++)r[a].isDataTexture?s.push(Ts(r[a].image)):s.push(Ts(r[a]))}else s=Ts(r);n.url=s}return t||(e.images[this.uuid]=n),n}}function Ts(i){return typeof HTMLImageElement<"u"&&i instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&i instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&i instanceof ImageBitmap?zu.getDataURL(i):i.data?{data:Array.from(i.data),width:i.width,height:i.height,type:i.data.constructor.name}:(et("Texture: Unable to serialize Texture."),{})}let Hu=0;const As=new $;class rn extends qi{constructor(e=rn.DEFAULT_IMAGE,t=rn.DEFAULT_MAPPING,n=Xn,r=Xn,s=Qt,a=xi,o=En,u=dn,l=rn.DEFAULT_ANISOTROPY,h=ri){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:Hu++}),this.uuid=gr(),this.name="",this.source=new ro(e),this.mipmaps=[],this.mapping=t,this.channel=0,this.wrapS=n,this.wrapT=r,this.magFilter=s,this.minFilter=a,this.anisotropy=l,this.format=o,this.internalFormat=null,this.type=u,this.offset=new pt(0,0),this.repeat=new pt(1,1),this.center=new pt(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new st,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=h,this.userData={},this.updateRanges=[],this.version=0,this.onUpdate=null,this.renderTarget=null,this.isRenderTargetTexture=!1,this.isArrayTexture=!!(e&&e.depth&&e.depth>1),this.pmremVersion=0}get width(){return this.source.getSize(As).x}get height(){return this.source.getSize(As).y}get depth(){return this.source.getSize(As).z}get image(){return this.source.data}set image(e=null){this.source.data=e}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}clone(){return new this.constructor().copy(this)}copy(e){return this.name=e.name,this.source=e.source,this.mipmaps=e.mipmaps.slice(0),this.mapping=e.mapping,this.channel=e.channel,this.wrapS=e.wrapS,this.wrapT=e.wrapT,this.magFilter=e.magFilter,this.minFilter=e.minFilter,this.anisotropy=e.anisotropy,this.format=e.format,this.internalFormat=e.internalFormat,this.type=e.type,this.offset.copy(e.offset),this.repeat.copy(e.repeat),this.center.copy(e.center),this.rotation=e.rotation,this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrix.copy(e.matrix),this.generateMipmaps=e.generateMipmaps,this.premultiplyAlpha=e.premultiplyAlpha,this.flipY=e.flipY,this.unpackAlignment=e.unpackAlignment,this.colorSpace=e.colorSpace,this.renderTarget=e.renderTarget,this.isRenderTargetTexture=e.isRenderTargetTexture,this.isArrayTexture=e.isArrayTexture,this.userData=JSON.parse(JSON.stringify(e.userData)),this.needsUpdate=!0,this}setValues(e){for(const t in e){const n=e[t];if(n===void 0){et(`Texture.setValues(): parameter '${t}' has value of undefined.`);continue}const r=this[t];if(r===void 0){et(`Texture.setValues(): property '${t}' does not exist.`);continue}r&&n&&r.isVector2&&n.isVector2||r&&n&&r.isVector3&&n.isVector3||r&&n&&r.isMatrix3&&n.isMatrix3?r.copy(n):this[t]=n}}toJSON(e){const t=e===void 0||typeof e=="string";if(!t&&e.textures[this.uuid]!==void 0)return e.textures[this.uuid];const n={metadata:{version:4.7,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(e).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(n.userData=this.userData),t||(e.textures[this.uuid]=n),n}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(e){if(this.mapping!==Cl)return e;if(e.applyMatrix3(this.matrix),e.x<0||e.x>1)switch(this.wrapS){case ca:e.x=e.x-Math.floor(e.x);break;case Xn:e.x=e.x<0?0:1;break;case ua:Math.abs(Math.floor(e.x)%2)===1?e.x=Math.ceil(e.x)-e.x:e.x=e.x-Math.floor(e.x);break}if(e.y<0||e.y>1)switch(this.wrapT){case ca:e.y=e.y-Math.floor(e.y);break;case Xn:e.y=e.y<0?0:1;break;case ua:Math.abs(Math.floor(e.y)%2)===1?e.y=Math.ceil(e.y)-e.y:e.y=e.y-Math.floor(e.y);break}return this.flipY&&(e.y=1-e.y),e}set needsUpdate(e){e===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(e){e===!0&&this.pmremVersion++}}rn.DEFAULT_IMAGE=null;rn.DEFAULT_MAPPING=Cl;rn.DEFAULT_ANISOTROPY=1;class Nt{constructor(e=0,t=0,n=0,r=1){Nt.prototype.isVector4=!0,this.x=e,this.y=t,this.z=n,this.w=r}get width(){return this.z}set width(e){this.z=e}get height(){return this.w}set height(e){this.w=e}set(e,t,n,r){return this.x=e,this.y=t,this.z=n,this.w=r,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this.w=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setW(e){return this.w=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;case 3:this.w=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this.w=e.w!==void 0?e.w:1,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this.w+=e.w,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this.w+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this.w=e.w+t.w,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this.w+=e.w*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this.w-=e.w,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this.w-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this.w=e.w-t.w,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this.w*=e.w,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this.w*=e,this}applyMatrix4(e){const t=this.x,n=this.y,r=this.z,s=this.w,a=e.elements;return this.x=a[0]*t+a[4]*n+a[8]*r+a[12]*s,this.y=a[1]*t+a[5]*n+a[9]*r+a[13]*s,this.z=a[2]*t+a[6]*n+a[10]*r+a[14]*s,this.w=a[3]*t+a[7]*n+a[11]*r+a[15]*s,this}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this.w/=e.w,this}divideScalar(e){return this.multiplyScalar(1/e)}setAxisAngleFromQuaternion(e){this.w=2*Math.acos(e.w);const t=Math.sqrt(1-e.w*e.w);return t<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=e.x/t,this.y=e.y/t,this.z=e.z/t),this}setAxisAngleFromRotationMatrix(e){let t,n,r,s;const u=e.elements,l=u[0],h=u[4],p=u[8],m=u[1],v=u[5],y=u[9],b=u[2],g=u[6],d=u[10];if(Math.abs(h-m)<.01&&Math.abs(p-b)<.01&&Math.abs(y-g)<.01){if(Math.abs(h+m)<.1&&Math.abs(p+b)<.1&&Math.abs(y+g)<.1&&Math.abs(l+v+d-3)<.1)return this.set(1,0,0,0),this;t=Math.PI;const C=(l+1)/2,A=(v+1)/2,P=(d+1)/2,D=(h+m)/4,I=(p+b)/4,G=(y+g)/4;return C>A&&C>P?C<.01?(n=0,r=.707106781,s=.707106781):(n=Math.sqrt(C),r=D/n,s=I/n):A>P?A<.01?(n=.707106781,r=0,s=.707106781):(r=Math.sqrt(A),n=D/r,s=G/r):P<.01?(n=.707106781,r=.707106781,s=0):(s=Math.sqrt(P),n=I/s,r=G/s),this.set(n,r,s,t),this}let R=Math.sqrt((g-y)*(g-y)+(p-b)*(p-b)+(m-h)*(m-h));return Math.abs(R)<.001&&(R=1),this.x=(g-y)/R,this.y=(p-b)/R,this.z=(m-h)/R,this.w=Math.acos((l+v+d-1)/2),this}setFromMatrixPosition(e){const t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this.w=t[15],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this.w=Math.min(this.w,e.w),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this.w=Math.max(this.w,e.w),this}clamp(e,t){return this.x=dt(this.x,e.x,t.x),this.y=dt(this.y,e.y,t.y),this.z=dt(this.z,e.z,t.z),this.w=dt(this.w,e.w,t.w),this}clampScalar(e,t){return this.x=dt(this.x,e,t),this.y=dt(this.y,e,t),this.z=dt(this.z,e,t),this.w=dt(this.w,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(dt(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z+this.w*e.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this.w+=(e.w-this.w)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this.z=e.z+(t.z-e.z)*n,this.w=e.w+(t.w-e.w)*n,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z&&e.w===this.w}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this.w=e[t+3],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e[t+3]=this.w,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this.w=e.getW(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}}class ku extends qi{constructor(e=1,t=1,n={}){super(),n=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:Qt,depthBuffer:!0,stencilBuffer:!1,resolveDepthBuffer:!0,resolveStencilBuffer:!0,depthTexture:null,samples:0,count:1,depth:1,multiview:!1},n),this.isRenderTarget=!0,this.width=e,this.height=t,this.depth=n.depth,this.scissor=new Nt(0,0,e,t),this.scissorTest=!1,this.viewport=new Nt(0,0,e,t);const r={width:e,height:t,depth:n.depth},s=new rn(r);this.textures=[];const a=n.count;for(let o=0;o<a;o++)this.textures[o]=s.clone(),this.textures[o].isRenderTargetTexture=!0,this.textures[o].renderTarget=this;this._setTextureOptions(n),this.depthBuffer=n.depthBuffer,this.stencilBuffer=n.stencilBuffer,this.resolveDepthBuffer=n.resolveDepthBuffer,this.resolveStencilBuffer=n.resolveStencilBuffer,this._depthTexture=null,this.depthTexture=n.depthTexture,this.samples=n.samples,this.multiview=n.multiview}_setTextureOptions(e={}){const t={minFilter:Qt,generateMipmaps:!1,flipY:!1,internalFormat:null};e.mapping!==void 0&&(t.mapping=e.mapping),e.wrapS!==void 0&&(t.wrapS=e.wrapS),e.wrapT!==void 0&&(t.wrapT=e.wrapT),e.wrapR!==void 0&&(t.wrapR=e.wrapR),e.magFilter!==void 0&&(t.magFilter=e.magFilter),e.minFilter!==void 0&&(t.minFilter=e.minFilter),e.format!==void 0&&(t.format=e.format),e.type!==void 0&&(t.type=e.type),e.anisotropy!==void 0&&(t.anisotropy=e.anisotropy),e.colorSpace!==void 0&&(t.colorSpace=e.colorSpace),e.flipY!==void 0&&(t.flipY=e.flipY),e.generateMipmaps!==void 0&&(t.generateMipmaps=e.generateMipmaps),e.internalFormat!==void 0&&(t.internalFormat=e.internalFormat);for(let n=0;n<this.textures.length;n++)this.textures[n].setValues(t)}get texture(){return this.textures[0]}set texture(e){this.textures[0]=e}set depthTexture(e){this._depthTexture!==null&&(this._depthTexture.renderTarget=null),e!==null&&(e.renderTarget=this),this._depthTexture=e}get depthTexture(){return this._depthTexture}setSize(e,t,n=1){if(this.width!==e||this.height!==t||this.depth!==n){this.width=e,this.height=t,this.depth=n;for(let r=0,s=this.textures.length;r<s;r++)this.textures[r].image.width=e,this.textures[r].image.height=t,this.textures[r].image.depth=n,this.textures[r].isData3DTexture!==!0&&(this.textures[r].isArrayTexture=this.textures[r].image.depth>1);this.dispose()}this.viewport.set(0,0,e,t),this.scissor.set(0,0,e,t)}clone(){return new this.constructor().copy(this)}copy(e){this.width=e.width,this.height=e.height,this.depth=e.depth,this.scissor.copy(e.scissor),this.scissorTest=e.scissorTest,this.viewport.copy(e.viewport),this.textures.length=0;for(let t=0,n=e.textures.length;t<n;t++){this.textures[t]=e.textures[t].clone(),this.textures[t].isRenderTargetTexture=!0,this.textures[t].renderTarget=this;const r=Object.assign({},e.textures[t].image);this.textures[t].source=new ro(r)}return this.depthBuffer=e.depthBuffer,this.stencilBuffer=e.stencilBuffer,this.resolveDepthBuffer=e.resolveDepthBuffer,this.resolveStencilBuffer=e.resolveStencilBuffer,e.depthTexture!==null&&(this.depthTexture=e.depthTexture.clone()),this.samples=e.samples,this}dispose(){this.dispatchEvent({type:"dispose"})}}class Dn extends ku{constructor(e=1,t=1,n={}){super(e,t,n),this.isWebGLRenderTarget=!0}}class Bl extends rn{constructor(e=null,t=1,n=1,r=1){super(null),this.isDataArrayTexture=!0,this.image={data:e,width:t,height:n,depth:r},this.magFilter=Xt,this.minFilter=Xt,this.wrapR=Xn,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}addLayerUpdate(e){this.layerUpdates.add(e)}clearLayerUpdates(){this.layerUpdates.clear()}}class Wu extends rn{constructor(e=null,t=1,n=1,r=1){super(null),this.isData3DTexture=!0,this.image={data:e,width:t,height:n,depth:r},this.magFilter=Xt,this.minFilter=Xt,this.wrapR=Xn,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class xr{constructor(e=new $(1/0,1/0,1/0),t=new $(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=e,this.max=t}set(e,t){return this.min.copy(e),this.max.copy(t),this}setFromArray(e){this.makeEmpty();for(let t=0,n=e.length;t<n;t+=3)this.expandByPoint(xn.fromArray(e,t));return this}setFromBufferAttribute(e){this.makeEmpty();for(let t=0,n=e.count;t<n;t++)this.expandByPoint(xn.fromBufferAttribute(e,t));return this}setFromPoints(e){this.makeEmpty();for(let t=0,n=e.length;t<n;t++)this.expandByPoint(e[t]);return this}setFromCenterAndSize(e,t){const n=xn.copy(t).multiplyScalar(.5);return this.min.copy(e).sub(n),this.max.copy(e).add(n),this}setFromObject(e,t=!1){return this.makeEmpty(),this.expandByObject(e,t)}clone(){return new this.constructor().copy(this)}copy(e){return this.min.copy(e.min),this.max.copy(e.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(e){return this.isEmpty()?e.set(0,0,0):e.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(e){return this.isEmpty()?e.set(0,0,0):e.subVectors(this.max,this.min)}expandByPoint(e){return this.min.min(e),this.max.max(e),this}expandByVector(e){return this.min.sub(e),this.max.add(e),this}expandByScalar(e){return this.min.addScalar(-e),this.max.addScalar(e),this}expandByObject(e,t=!1){e.updateWorldMatrix(!1,!1);const n=e.geometry;if(n!==void 0){const s=n.getAttribute("position");if(t===!0&&s!==void 0&&e.isInstancedMesh!==!0)for(let a=0,o=s.count;a<o;a++)e.isMesh===!0?e.getVertexPosition(a,xn):xn.fromBufferAttribute(s,a),xn.applyMatrix4(e.matrixWorld),this.expandByPoint(xn);else e.boundingBox!==void 0?(e.boundingBox===null&&e.computeBoundingBox(),Pr.copy(e.boundingBox)):(n.boundingBox===null&&n.computeBoundingBox(),Pr.copy(n.boundingBox)),Pr.applyMatrix4(e.matrixWorld),this.union(Pr)}const r=e.children;for(let s=0,a=r.length;s<a;s++)this.expandByObject(r[s],t);return this}containsPoint(e){return e.x>=this.min.x&&e.x<=this.max.x&&e.y>=this.min.y&&e.y<=this.max.y&&e.z>=this.min.z&&e.z<=this.max.z}containsBox(e){return this.min.x<=e.min.x&&e.max.x<=this.max.x&&this.min.y<=e.min.y&&e.max.y<=this.max.y&&this.min.z<=e.min.z&&e.max.z<=this.max.z}getParameter(e,t){return t.set((e.x-this.min.x)/(this.max.x-this.min.x),(e.y-this.min.y)/(this.max.y-this.min.y),(e.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(e){return e.max.x>=this.min.x&&e.min.x<=this.max.x&&e.max.y>=this.min.y&&e.min.y<=this.max.y&&e.max.z>=this.min.z&&e.min.z<=this.max.z}intersectsSphere(e){return this.clampPoint(e.center,xn),xn.distanceToSquared(e.center)<=e.radius*e.radius}intersectsPlane(e){let t,n;return e.normal.x>0?(t=e.normal.x*this.min.x,n=e.normal.x*this.max.x):(t=e.normal.x*this.max.x,n=e.normal.x*this.min.x),e.normal.y>0?(t+=e.normal.y*this.min.y,n+=e.normal.y*this.max.y):(t+=e.normal.y*this.max.y,n+=e.normal.y*this.min.y),e.normal.z>0?(t+=e.normal.z*this.min.z,n+=e.normal.z*this.max.z):(t+=e.normal.z*this.max.z,n+=e.normal.z*this.min.z),t<=-e.constant&&n>=-e.constant}intersectsTriangle(e){if(this.isEmpty())return!1;this.getCenter(rr),Dr.subVectors(this.max,rr),Ri.subVectors(e.a,rr),Ci.subVectors(e.b,rr),Pi.subVectors(e.c,rr),Jn.subVectors(Ci,Ri),Qn.subVectors(Pi,Ci),ui.subVectors(Ri,Pi);let t=[0,-Jn.z,Jn.y,0,-Qn.z,Qn.y,0,-ui.z,ui.y,Jn.z,0,-Jn.x,Qn.z,0,-Qn.x,ui.z,0,-ui.x,-Jn.y,Jn.x,0,-Qn.y,Qn.x,0,-ui.y,ui.x,0];return!ws(t,Ri,Ci,Pi,Dr)||(t=[1,0,0,0,1,0,0,0,1],!ws(t,Ri,Ci,Pi,Dr))?!1:(Lr.crossVectors(Jn,Qn),t=[Lr.x,Lr.y,Lr.z],ws(t,Ri,Ci,Pi,Dr))}clampPoint(e,t){return t.copy(e).clamp(this.min,this.max)}distanceToPoint(e){return this.clampPoint(e,xn).distanceTo(e)}getBoundingSphere(e){return this.isEmpty()?e.makeEmpty():(this.getCenter(e.center),e.radius=this.getSize(xn).length()*.5),e}intersect(e){return this.min.max(e.min),this.max.min(e.max),this.isEmpty()&&this.makeEmpty(),this}union(e){return this.min.min(e.min),this.max.max(e.max),this}applyMatrix4(e){return this.isEmpty()?this:(Vn[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(e),Vn[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(e),Vn[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(e),Vn[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(e),Vn[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(e),Vn[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(e),Vn[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(e),Vn[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(e),this.setFromPoints(Vn),this)}translate(e){return this.min.add(e),this.max.add(e),this}equals(e){return e.min.equals(this.min)&&e.max.equals(this.max)}toJSON(){return{min:this.min.toArray(),max:this.max.toArray()}}fromJSON(e){return this.min.fromArray(e.min),this.max.fromArray(e.max),this}}const Vn=[new $,new $,new $,new $,new $,new $,new $,new $],xn=new $,Pr=new xr,Ri=new $,Ci=new $,Pi=new $,Jn=new $,Qn=new $,ui=new $,rr=new $,Dr=new $,Lr=new $,fi=new $;function ws(i,e,t,n,r){for(let s=0,a=i.length-3;s<=a;s+=3){fi.fromArray(i,s);const o=r.x*Math.abs(fi.x)+r.y*Math.abs(fi.y)+r.z*Math.abs(fi.z),u=e.dot(fi),l=t.dot(fi),h=n.dot(fi);if(Math.max(-Math.max(u,l,h),Math.min(u,l,h))>o)return!1}return!0}const Xu=new xr,sr=new $,Rs=new $;class so{constructor(e=new $,t=-1){this.isSphere=!0,this.center=e,this.radius=t}set(e,t){return this.center.copy(e),this.radius=t,this}setFromPoints(e,t){const n=this.center;t!==void 0?n.copy(t):Xu.setFromPoints(e).getCenter(n);let r=0;for(let s=0,a=e.length;s<a;s++)r=Math.max(r,n.distanceToSquared(e[s]));return this.radius=Math.sqrt(r),this}copy(e){return this.center.copy(e.center),this.radius=e.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(e){return e.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(e){return e.distanceTo(this.center)-this.radius}intersectsSphere(e){const t=this.radius+e.radius;return e.center.distanceToSquared(this.center)<=t*t}intersectsBox(e){return e.intersectsSphere(this)}intersectsPlane(e){return Math.abs(e.distanceToPoint(this.center))<=this.radius}clampPoint(e,t){const n=this.center.distanceToSquared(e);return t.copy(e),n>this.radius*this.radius&&(t.sub(this.center).normalize(),t.multiplyScalar(this.radius).add(this.center)),t}getBoundingBox(e){return this.isEmpty()?(e.makeEmpty(),e):(e.set(this.center,this.center),e.expandByScalar(this.radius),e)}applyMatrix4(e){return this.center.applyMatrix4(e),this.radius=this.radius*e.getMaxScaleOnAxis(),this}translate(e){return this.center.add(e),this}expandByPoint(e){if(this.isEmpty())return this.center.copy(e),this.radius=0,this;sr.subVectors(e,this.center);const t=sr.lengthSq();if(t>this.radius*this.radius){const n=Math.sqrt(t),r=(n-this.radius)*.5;this.center.addScaledVector(sr,r/n),this.radius+=r}return this}union(e){return e.isEmpty()?this:this.isEmpty()?(this.copy(e),this):(this.center.equals(e.center)===!0?this.radius=Math.max(this.radius,e.radius):(Rs.subVectors(e.center,this.center).setLength(e.radius),this.expandByPoint(sr.copy(e.center).add(Rs)),this.expandByPoint(sr.copy(e.center).sub(Rs))),this)}equals(e){return e.center.equals(this.center)&&e.radius===this.radius}clone(){return new this.constructor().copy(this)}toJSON(){return{radius:this.radius,center:this.center.toArray()}}fromJSON(e){return this.radius=e.radius,this.center.fromArray(e.center),this}}const zn=new $,Cs=new $,Ur=new $,ei=new $,Ps=new $,Ir=new $,Ds=new $;class $u{constructor(e=new $,t=new $(0,0,-1)){this.origin=e,this.direction=t}set(e,t){return this.origin.copy(e),this.direction.copy(t),this}copy(e){return this.origin.copy(e.origin),this.direction.copy(e.direction),this}at(e,t){return t.copy(this.origin).addScaledVector(this.direction,e)}lookAt(e){return this.direction.copy(e).sub(this.origin).normalize(),this}recast(e){return this.origin.copy(this.at(e,zn)),this}closestPointToPoint(e,t){t.subVectors(e,this.origin);const n=t.dot(this.direction);return n<0?t.copy(this.origin):t.copy(this.origin).addScaledVector(this.direction,n)}distanceToPoint(e){return Math.sqrt(this.distanceSqToPoint(e))}distanceSqToPoint(e){const t=zn.subVectors(e,this.origin).dot(this.direction);return t<0?this.origin.distanceToSquared(e):(zn.copy(this.origin).addScaledVector(this.direction,t),zn.distanceToSquared(e))}distanceSqToSegment(e,t,n,r){Cs.copy(e).add(t).multiplyScalar(.5),Ur.copy(t).sub(e).normalize(),ei.copy(this.origin).sub(Cs);const s=e.distanceTo(t)*.5,a=-this.direction.dot(Ur),o=ei.dot(this.direction),u=-ei.dot(Ur),l=ei.lengthSq(),h=Math.abs(1-a*a);let p,m,v,y;if(h>0)if(p=a*u-o,m=a*o-u,y=s*h,p>=0)if(m>=-y)if(m<=y){const b=1/h;p*=b,m*=b,v=p*(p+a*m+2*o)+m*(a*p+m+2*u)+l}else m=s,p=Math.max(0,-(a*m+o)),v=-p*p+m*(m+2*u)+l;else m=-s,p=Math.max(0,-(a*m+o)),v=-p*p+m*(m+2*u)+l;else m<=-y?(p=Math.max(0,-(-a*s+o)),m=p>0?-s:Math.min(Math.max(-s,-u),s),v=-p*p+m*(m+2*u)+l):m<=y?(p=0,m=Math.min(Math.max(-s,-u),s),v=m*(m+2*u)+l):(p=Math.max(0,-(a*s+o)),m=p>0?s:Math.min(Math.max(-s,-u),s),v=-p*p+m*(m+2*u)+l);else m=a>0?-s:s,p=Math.max(0,-(a*m+o)),v=-p*p+m*(m+2*u)+l;return n&&n.copy(this.origin).addScaledVector(this.direction,p),r&&r.copy(Cs).addScaledVector(Ur,m),v}intersectSphere(e,t){zn.subVectors(e.center,this.origin);const n=zn.dot(this.direction),r=zn.dot(zn)-n*n,s=e.radius*e.radius;if(r>s)return null;const a=Math.sqrt(s-r),o=n-a,u=n+a;return u<0?null:o<0?this.at(u,t):this.at(o,t)}intersectsSphere(e){return e.radius<0?!1:this.distanceSqToPoint(e.center)<=e.radius*e.radius}distanceToPlane(e){const t=e.normal.dot(this.direction);if(t===0)return e.distanceToPoint(this.origin)===0?0:null;const n=-(this.origin.dot(e.normal)+e.constant)/t;return n>=0?n:null}intersectPlane(e,t){const n=this.distanceToPlane(e);return n===null?null:this.at(n,t)}intersectsPlane(e){const t=e.distanceToPoint(this.origin);return t===0||e.normal.dot(this.direction)*t<0}intersectBox(e,t){let n,r,s,a,o,u;const l=1/this.direction.x,h=1/this.direction.y,p=1/this.direction.z,m=this.origin;return l>=0?(n=(e.min.x-m.x)*l,r=(e.max.x-m.x)*l):(n=(e.max.x-m.x)*l,r=(e.min.x-m.x)*l),h>=0?(s=(e.min.y-m.y)*h,a=(e.max.y-m.y)*h):(s=(e.max.y-m.y)*h,a=(e.min.y-m.y)*h),n>a||s>r||((s>n||isNaN(n))&&(n=s),(a<r||isNaN(r))&&(r=a),p>=0?(o=(e.min.z-m.z)*p,u=(e.max.z-m.z)*p):(o=(e.max.z-m.z)*p,u=(e.min.z-m.z)*p),n>u||o>r)||((o>n||n!==n)&&(n=o),(u<r||r!==r)&&(r=u),r<0)?null:this.at(n>=0?n:r,t)}intersectsBox(e){return this.intersectBox(e,zn)!==null}intersectTriangle(e,t,n,r,s){Ps.subVectors(t,e),Ir.subVectors(n,e),Ds.crossVectors(Ps,Ir);let a=this.direction.dot(Ds),o;if(a>0){if(r)return null;o=1}else if(a<0)o=-1,a=-a;else return null;ei.subVectors(this.origin,e);const u=o*this.direction.dot(Ir.crossVectors(ei,Ir));if(u<0)return null;const l=o*this.direction.dot(Ps.cross(ei));if(l<0||u+l>a)return null;const h=-o*ei.dot(Ds);return h<0?null:this.at(h/a,s)}applyMatrix4(e){return this.origin.applyMatrix4(e),this.direction.transformDirection(e),this}equals(e){return e.origin.equals(this.origin)&&e.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}}class Ot{constructor(e,t,n,r,s,a,o,u,l,h,p,m,v,y,b,g){Ot.prototype.isMatrix4=!0,this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],e!==void 0&&this.set(e,t,n,r,s,a,o,u,l,h,p,m,v,y,b,g)}set(e,t,n,r,s,a,o,u,l,h,p,m,v,y,b,g){const d=this.elements;return d[0]=e,d[4]=t,d[8]=n,d[12]=r,d[1]=s,d[5]=a,d[9]=o,d[13]=u,d[2]=l,d[6]=h,d[10]=p,d[14]=m,d[3]=v,d[7]=y,d[11]=b,d[15]=g,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new Ot().fromArray(this.elements)}copy(e){const t=this.elements,n=e.elements;return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=n[3],t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=n[7],t[8]=n[8],t[9]=n[9],t[10]=n[10],t[11]=n[11],t[12]=n[12],t[13]=n[13],t[14]=n[14],t[15]=n[15],this}copyPosition(e){const t=this.elements,n=e.elements;return t[12]=n[12],t[13]=n[13],t[14]=n[14],this}setFromMatrix3(e){const t=e.elements;return this.set(t[0],t[3],t[6],0,t[1],t[4],t[7],0,t[2],t[5],t[8],0,0,0,0,1),this}extractBasis(e,t,n){return this.determinant()===0?(e.set(1,0,0),t.set(0,1,0),n.set(0,0,1),this):(e.setFromMatrixColumn(this,0),t.setFromMatrixColumn(this,1),n.setFromMatrixColumn(this,2),this)}makeBasis(e,t,n){return this.set(e.x,t.x,n.x,0,e.y,t.y,n.y,0,e.z,t.z,n.z,0,0,0,0,1),this}extractRotation(e){if(e.determinant()===0)return this.identity();const t=this.elements,n=e.elements,r=1/Di.setFromMatrixColumn(e,0).length(),s=1/Di.setFromMatrixColumn(e,1).length(),a=1/Di.setFromMatrixColumn(e,2).length();return t[0]=n[0]*r,t[1]=n[1]*r,t[2]=n[2]*r,t[3]=0,t[4]=n[4]*s,t[5]=n[5]*s,t[6]=n[6]*s,t[7]=0,t[8]=n[8]*a,t[9]=n[9]*a,t[10]=n[10]*a,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromEuler(e){const t=this.elements,n=e.x,r=e.y,s=e.z,a=Math.cos(n),o=Math.sin(n),u=Math.cos(r),l=Math.sin(r),h=Math.cos(s),p=Math.sin(s);if(e.order==="XYZ"){const m=a*h,v=a*p,y=o*h,b=o*p;t[0]=u*h,t[4]=-u*p,t[8]=l,t[1]=v+y*l,t[5]=m-b*l,t[9]=-o*u,t[2]=b-m*l,t[6]=y+v*l,t[10]=a*u}else if(e.order==="YXZ"){const m=u*h,v=u*p,y=l*h,b=l*p;t[0]=m+b*o,t[4]=y*o-v,t[8]=a*l,t[1]=a*p,t[5]=a*h,t[9]=-o,t[2]=v*o-y,t[6]=b+m*o,t[10]=a*u}else if(e.order==="ZXY"){const m=u*h,v=u*p,y=l*h,b=l*p;t[0]=m-b*o,t[4]=-a*p,t[8]=y+v*o,t[1]=v+y*o,t[5]=a*h,t[9]=b-m*o,t[2]=-a*l,t[6]=o,t[10]=a*u}else if(e.order==="ZYX"){const m=a*h,v=a*p,y=o*h,b=o*p;t[0]=u*h,t[4]=y*l-v,t[8]=m*l+b,t[1]=u*p,t[5]=b*l+m,t[9]=v*l-y,t[2]=-l,t[6]=o*u,t[10]=a*u}else if(e.order==="YZX"){const m=a*u,v=a*l,y=o*u,b=o*l;t[0]=u*h,t[4]=b-m*p,t[8]=y*p+v,t[1]=p,t[5]=a*h,t[9]=-o*h,t[2]=-l*h,t[6]=v*p+y,t[10]=m-b*p}else if(e.order==="XZY"){const m=a*u,v=a*l,y=o*u,b=o*l;t[0]=u*h,t[4]=-p,t[8]=l*h,t[1]=m*p+b,t[5]=a*h,t[9]=v*p-y,t[2]=y*p-v,t[6]=o*h,t[10]=b*p+m}return t[3]=0,t[7]=0,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromQuaternion(e){return this.compose(qu,e,Yu)}lookAt(e,t,n){const r=this.elements;return un.subVectors(e,t),un.lengthSq()===0&&(un.z=1),un.normalize(),ti.crossVectors(n,un),ti.lengthSq()===0&&(Math.abs(n.z)===1?un.x+=1e-4:un.z+=1e-4,un.normalize(),ti.crossVectors(n,un)),ti.normalize(),Fr.crossVectors(un,ti),r[0]=ti.x,r[4]=Fr.x,r[8]=un.x,r[1]=ti.y,r[5]=Fr.y,r[9]=un.y,r[2]=ti.z,r[6]=Fr.z,r[10]=un.z,this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){const n=e.elements,r=t.elements,s=this.elements,a=n[0],o=n[4],u=n[8],l=n[12],h=n[1],p=n[5],m=n[9],v=n[13],y=n[2],b=n[6],g=n[10],d=n[14],R=n[3],C=n[7],A=n[11],P=n[15],D=r[0],I=r[4],G=r[8],x=r[12],E=r[1],B=r[5],j=r[9],K=r[13],ie=r[2],oe=r[6],Q=r[10],te=r[14],ue=r[3],Ae=r[7],Ee=r[11],Le=r[15];return s[0]=a*D+o*E+u*ie+l*ue,s[4]=a*I+o*B+u*oe+l*Ae,s[8]=a*G+o*j+u*Q+l*Ee,s[12]=a*x+o*K+u*te+l*Le,s[1]=h*D+p*E+m*ie+v*ue,s[5]=h*I+p*B+m*oe+v*Ae,s[9]=h*G+p*j+m*Q+v*Ee,s[13]=h*x+p*K+m*te+v*Le,s[2]=y*D+b*E+g*ie+d*ue,s[6]=y*I+b*B+g*oe+d*Ae,s[10]=y*G+b*j+g*Q+d*Ee,s[14]=y*x+b*K+g*te+d*Le,s[3]=R*D+C*E+A*ie+P*ue,s[7]=R*I+C*B+A*oe+P*Ae,s[11]=R*G+C*j+A*Q+P*Ee,s[15]=R*x+C*K+A*te+P*Le,this}multiplyScalar(e){const t=this.elements;return t[0]*=e,t[4]*=e,t[8]*=e,t[12]*=e,t[1]*=e,t[5]*=e,t[9]*=e,t[13]*=e,t[2]*=e,t[6]*=e,t[10]*=e,t[14]*=e,t[3]*=e,t[7]*=e,t[11]*=e,t[15]*=e,this}determinant(){const e=this.elements,t=e[0],n=e[4],r=e[8],s=e[12],a=e[1],o=e[5],u=e[9],l=e[13],h=e[2],p=e[6],m=e[10],v=e[14],y=e[3],b=e[7],g=e[11],d=e[15],R=u*v-l*m,C=o*v-l*p,A=o*m-u*p,P=a*v-l*h,D=a*m-u*h,I=a*p-o*h;return t*(b*R-g*C+d*A)-n*(y*R-g*P+d*D)+r*(y*C-b*P+d*I)-s*(y*A-b*D+g*I)}transpose(){const e=this.elements;let t;return t=e[1],e[1]=e[4],e[4]=t,t=e[2],e[2]=e[8],e[8]=t,t=e[6],e[6]=e[9],e[9]=t,t=e[3],e[3]=e[12],e[12]=t,t=e[7],e[7]=e[13],e[13]=t,t=e[11],e[11]=e[14],e[14]=t,this}setPosition(e,t,n){const r=this.elements;return e.isVector3?(r[12]=e.x,r[13]=e.y,r[14]=e.z):(r[12]=e,r[13]=t,r[14]=n),this}invert(){const e=this.elements,t=e[0],n=e[1],r=e[2],s=e[3],a=e[4],o=e[5],u=e[6],l=e[7],h=e[8],p=e[9],m=e[10],v=e[11],y=e[12],b=e[13],g=e[14],d=e[15],R=p*g*l-b*m*l+b*u*v-o*g*v-p*u*d+o*m*d,C=y*m*l-h*g*l-y*u*v+a*g*v+h*u*d-a*m*d,A=h*b*l-y*p*l+y*o*v-a*b*v-h*o*d+a*p*d,P=y*p*u-h*b*u-y*o*m+a*b*m+h*o*g-a*p*g,D=t*R+n*C+r*A+s*P;if(D===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);const I=1/D;return e[0]=R*I,e[1]=(b*m*s-p*g*s-b*r*v+n*g*v+p*r*d-n*m*d)*I,e[2]=(o*g*s-b*u*s+b*r*l-n*g*l-o*r*d+n*u*d)*I,e[3]=(p*u*s-o*m*s-p*r*l+n*m*l+o*r*v-n*u*v)*I,e[4]=C*I,e[5]=(h*g*s-y*m*s+y*r*v-t*g*v-h*r*d+t*m*d)*I,e[6]=(y*u*s-a*g*s-y*r*l+t*g*l+a*r*d-t*u*d)*I,e[7]=(a*m*s-h*u*s+h*r*l-t*m*l-a*r*v+t*u*v)*I,e[8]=A*I,e[9]=(y*p*s-h*b*s-y*n*v+t*b*v+h*n*d-t*p*d)*I,e[10]=(a*b*s-y*o*s+y*n*l-t*b*l-a*n*d+t*o*d)*I,e[11]=(h*o*s-a*p*s-h*n*l+t*p*l+a*n*v-t*o*v)*I,e[12]=P*I,e[13]=(h*b*r-y*p*r+y*n*m-t*b*m-h*n*g+t*p*g)*I,e[14]=(y*o*r-a*b*r-y*n*u+t*b*u+a*n*g-t*o*g)*I,e[15]=(a*p*r-h*o*r+h*n*u-t*p*u-a*n*m+t*o*m)*I,this}scale(e){const t=this.elements,n=e.x,r=e.y,s=e.z;return t[0]*=n,t[4]*=r,t[8]*=s,t[1]*=n,t[5]*=r,t[9]*=s,t[2]*=n,t[6]*=r,t[10]*=s,t[3]*=n,t[7]*=r,t[11]*=s,this}getMaxScaleOnAxis(){const e=this.elements,t=e[0]*e[0]+e[1]*e[1]+e[2]*e[2],n=e[4]*e[4]+e[5]*e[5]+e[6]*e[6],r=e[8]*e[8]+e[9]*e[9]+e[10]*e[10];return Math.sqrt(Math.max(t,n,r))}makeTranslation(e,t,n){return e.isVector3?this.set(1,0,0,e.x,0,1,0,e.y,0,0,1,e.z,0,0,0,1):this.set(1,0,0,e,0,1,0,t,0,0,1,n,0,0,0,1),this}makeRotationX(e){const t=Math.cos(e),n=Math.sin(e);return this.set(1,0,0,0,0,t,-n,0,0,n,t,0,0,0,0,1),this}makeRotationY(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,0,n,0,0,1,0,0,-n,0,t,0,0,0,0,1),this}makeRotationZ(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,-n,0,0,n,t,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(e,t){const n=Math.cos(t),r=Math.sin(t),s=1-n,a=e.x,o=e.y,u=e.z,l=s*a,h=s*o;return this.set(l*a+n,l*o-r*u,l*u+r*o,0,l*o+r*u,h*o+n,h*u-r*a,0,l*u-r*o,h*u+r*a,s*u*u+n,0,0,0,0,1),this}makeScale(e,t,n){return this.set(e,0,0,0,0,t,0,0,0,0,n,0,0,0,0,1),this}makeShear(e,t,n,r,s,a){return this.set(1,n,s,0,e,1,a,0,t,r,1,0,0,0,0,1),this}compose(e,t,n){const r=this.elements,s=t._x,a=t._y,o=t._z,u=t._w,l=s+s,h=a+a,p=o+o,m=s*l,v=s*h,y=s*p,b=a*h,g=a*p,d=o*p,R=u*l,C=u*h,A=u*p,P=n.x,D=n.y,I=n.z;return r[0]=(1-(b+d))*P,r[1]=(v+A)*P,r[2]=(y-C)*P,r[3]=0,r[4]=(v-A)*D,r[5]=(1-(m+d))*D,r[6]=(g+R)*D,r[7]=0,r[8]=(y+C)*I,r[9]=(g-R)*I,r[10]=(1-(m+b))*I,r[11]=0,r[12]=e.x,r[13]=e.y,r[14]=e.z,r[15]=1,this}decompose(e,t,n){const r=this.elements;if(e.x=r[12],e.y=r[13],e.z=r[14],this.determinant()===0)return n.set(1,1,1),t.identity(),this;let s=Di.set(r[0],r[1],r[2]).length();const a=Di.set(r[4],r[5],r[6]).length(),o=Di.set(r[8],r[9],r[10]).length();this.determinant()<0&&(s=-s),Mn.copy(this);const l=1/s,h=1/a,p=1/o;return Mn.elements[0]*=l,Mn.elements[1]*=l,Mn.elements[2]*=l,Mn.elements[4]*=h,Mn.elements[5]*=h,Mn.elements[6]*=h,Mn.elements[8]*=p,Mn.elements[9]*=p,Mn.elements[10]*=p,t.setFromRotationMatrix(Mn),n.x=s,n.y=a,n.z=o,this}makePerspective(e,t,n,r,s,a,o=Cn,u=!1){const l=this.elements,h=2*s/(t-e),p=2*s/(n-r),m=(t+e)/(t-e),v=(n+r)/(n-r);let y,b;if(u)y=s/(a-s),b=a*s/(a-s);else if(o===Cn)y=-(a+s)/(a-s),b=-2*a*s/(a-s);else if(o===is)y=-a/(a-s),b=-a*s/(a-s);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+o);return l[0]=h,l[4]=0,l[8]=m,l[12]=0,l[1]=0,l[5]=p,l[9]=v,l[13]=0,l[2]=0,l[6]=0,l[10]=y,l[14]=b,l[3]=0,l[7]=0,l[11]=-1,l[15]=0,this}makeOrthographic(e,t,n,r,s,a,o=Cn,u=!1){const l=this.elements,h=2/(t-e),p=2/(n-r),m=-(t+e)/(t-e),v=-(n+r)/(n-r);let y,b;if(u)y=1/(a-s),b=a/(a-s);else if(o===Cn)y=-2/(a-s),b=-(a+s)/(a-s);else if(o===is)y=-1/(a-s),b=-s/(a-s);else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+o);return l[0]=h,l[4]=0,l[8]=0,l[12]=m,l[1]=0,l[5]=p,l[9]=0,l[13]=v,l[2]=0,l[6]=0,l[10]=y,l[14]=b,l[3]=0,l[7]=0,l[11]=0,l[15]=1,this}equals(e){const t=this.elements,n=e.elements;for(let r=0;r<16;r++)if(t[r]!==n[r])return!1;return!0}fromArray(e,t=0){for(let n=0;n<16;n++)this.elements[n]=e[n+t];return this}toArray(e=[],t=0){const n=this.elements;return e[t]=n[0],e[t+1]=n[1],e[t+2]=n[2],e[t+3]=n[3],e[t+4]=n[4],e[t+5]=n[5],e[t+6]=n[6],e[t+7]=n[7],e[t+8]=n[8],e[t+9]=n[9],e[t+10]=n[10],e[t+11]=n[11],e[t+12]=n[12],e[t+13]=n[13],e[t+14]=n[14],e[t+15]=n[15],e}}const Di=new $,Mn=new Ot,qu=new $(0,0,0),Yu=new $(1,1,1),ti=new $,Fr=new $,un=new $,Io=new Ot,Fo=new vr;class Un{constructor(e=0,t=0,n=0,r=Un.DEFAULT_ORDER){this.isEuler=!0,this._x=e,this._y=t,this._z=n,this._order=r}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get order(){return this._order}set order(e){this._order=e,this._onChangeCallback()}set(e,t,n,r=this._order){return this._x=e,this._y=t,this._z=n,this._order=r,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(e){return this._x=e._x,this._y=e._y,this._z=e._z,this._order=e._order,this._onChangeCallback(),this}setFromRotationMatrix(e,t=this._order,n=!0){const r=e.elements,s=r[0],a=r[4],o=r[8],u=r[1],l=r[5],h=r[9],p=r[2],m=r[6],v=r[10];switch(t){case"XYZ":this._y=Math.asin(dt(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(-h,v),this._z=Math.atan2(-a,s)):(this._x=Math.atan2(m,l),this._z=0);break;case"YXZ":this._x=Math.asin(-dt(h,-1,1)),Math.abs(h)<.9999999?(this._y=Math.atan2(o,v),this._z=Math.atan2(u,l)):(this._y=Math.atan2(-p,s),this._z=0);break;case"ZXY":this._x=Math.asin(dt(m,-1,1)),Math.abs(m)<.9999999?(this._y=Math.atan2(-p,v),this._z=Math.atan2(-a,l)):(this._y=0,this._z=Math.atan2(u,s));break;case"ZYX":this._y=Math.asin(-dt(p,-1,1)),Math.abs(p)<.9999999?(this._x=Math.atan2(m,v),this._z=Math.atan2(u,s)):(this._x=0,this._z=Math.atan2(-a,l));break;case"YZX":this._z=Math.asin(dt(u,-1,1)),Math.abs(u)<.9999999?(this._x=Math.atan2(-h,l),this._y=Math.atan2(-p,s)):(this._x=0,this._y=Math.atan2(o,v));break;case"XZY":this._z=Math.asin(-dt(a,-1,1)),Math.abs(a)<.9999999?(this._x=Math.atan2(m,l),this._y=Math.atan2(o,s)):(this._x=Math.atan2(-h,v),this._y=0);break;default:et("Euler: .setFromRotationMatrix() encountered an unknown order: "+t)}return this._order=t,n===!0&&this._onChangeCallback(),this}setFromQuaternion(e,t,n){return Io.makeRotationFromQuaternion(e),this.setFromRotationMatrix(Io,t,n)}setFromVector3(e,t=this._order){return this.set(e.x,e.y,e.z,t)}reorder(e){return Fo.setFromEuler(this),this.setFromQuaternion(Fo,e)}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._order===this._order}fromArray(e){return this._x=e[0],this._y=e[1],this._z=e[2],e[3]!==void 0&&(this._order=e[3]),this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._order,e}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}}Un.DEFAULT_ORDER="XYZ";class Vl{constructor(){this.mask=1}set(e){this.mask=(1<<e|0)>>>0}enable(e){this.mask|=1<<e|0}enableAll(){this.mask=-1}toggle(e){this.mask^=1<<e|0}disable(e){this.mask&=~(1<<e|0)}disableAll(){this.mask=0}test(e){return(this.mask&e.mask)!==0}isEnabled(e){return(this.mask&(1<<e|0))!==0}}let Ku=0;const No=new $,Li=new vr,Gn=new Ot,Nr=new $,ar=new $,ju=new $,Zu=new vr,Oo=new $(1,0,0),Bo=new $(0,1,0),Vo=new $(0,0,1),zo={type:"added"},Ju={type:"removed"},Ui={type:"childadded",child:null},Ls={type:"childremoved",child:null};class ln extends qi{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:Ku++}),this.uuid=gr(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=ln.DEFAULT_UP.clone();const e=new $,t=new Un,n=new vr,r=new $(1,1,1);function s(){n.setFromEuler(t,!1)}function a(){t.setFromQuaternion(n,void 0,!1)}t._onChange(s),n._onChange(a),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:e},rotation:{configurable:!0,enumerable:!0,value:t},quaternion:{configurable:!0,enumerable:!0,value:n},scale:{configurable:!0,enumerable:!0,value:r},modelViewMatrix:{value:new Ot},normalMatrix:{value:new st}}),this.matrix=new Ot,this.matrixWorld=new Ot,this.matrixAutoUpdate=ln.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=ln.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new Vl,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.customDepthMaterial=void 0,this.customDistanceMaterial=void 0,this.userData={}}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(e){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(e),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(e){return this.quaternion.premultiply(e),this}setRotationFromAxisAngle(e,t){this.quaternion.setFromAxisAngle(e,t)}setRotationFromEuler(e){this.quaternion.setFromEuler(e,!0)}setRotationFromMatrix(e){this.quaternion.setFromRotationMatrix(e)}setRotationFromQuaternion(e){this.quaternion.copy(e)}rotateOnAxis(e,t){return Li.setFromAxisAngle(e,t),this.quaternion.multiply(Li),this}rotateOnWorldAxis(e,t){return Li.setFromAxisAngle(e,t),this.quaternion.premultiply(Li),this}rotateX(e){return this.rotateOnAxis(Oo,e)}rotateY(e){return this.rotateOnAxis(Bo,e)}rotateZ(e){return this.rotateOnAxis(Vo,e)}translateOnAxis(e,t){return No.copy(e).applyQuaternion(this.quaternion),this.position.add(No.multiplyScalar(t)),this}translateX(e){return this.translateOnAxis(Oo,e)}translateY(e){return this.translateOnAxis(Bo,e)}translateZ(e){return this.translateOnAxis(Vo,e)}localToWorld(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(this.matrixWorld)}worldToLocal(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(Gn.copy(this.matrixWorld).invert())}lookAt(e,t,n){e.isVector3?Nr.copy(e):Nr.set(e,t,n);const r=this.parent;this.updateWorldMatrix(!0,!1),ar.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?Gn.lookAt(ar,Nr,this.up):Gn.lookAt(Nr,ar,this.up),this.quaternion.setFromRotationMatrix(Gn),r&&(Gn.extractRotation(r.matrixWorld),Li.setFromRotationMatrix(Gn),this.quaternion.premultiply(Li.invert()))}add(e){if(arguments.length>1){for(let t=0;t<arguments.length;t++)this.add(arguments[t]);return this}return e===this?(xt("Object3D.add: object can't be added as a child of itself.",e),this):(e&&e.isObject3D?(e.removeFromParent(),e.parent=this,this.children.push(e),e.dispatchEvent(zo),Ui.child=e,this.dispatchEvent(Ui),Ui.child=null):xt("Object3D.add: object not an instance of THREE.Object3D.",e),this)}remove(e){if(arguments.length>1){for(let n=0;n<arguments.length;n++)this.remove(arguments[n]);return this}const t=this.children.indexOf(e);return t!==-1&&(e.parent=null,this.children.splice(t,1),e.dispatchEvent(Ju),Ls.child=e,this.dispatchEvent(Ls),Ls.child=null),this}removeFromParent(){const e=this.parent;return e!==null&&e.remove(this),this}clear(){return this.remove(...this.children)}attach(e){return this.updateWorldMatrix(!0,!1),Gn.copy(this.matrixWorld).invert(),e.parent!==null&&(e.parent.updateWorldMatrix(!0,!1),Gn.multiply(e.parent.matrixWorld)),e.applyMatrix4(Gn),e.removeFromParent(),e.parent=this,this.children.push(e),e.updateWorldMatrix(!1,!0),e.dispatchEvent(zo),Ui.child=e,this.dispatchEvent(Ui),Ui.child=null,this}getObjectById(e){return this.getObjectByProperty("id",e)}getObjectByName(e){return this.getObjectByProperty("name",e)}getObjectByProperty(e,t){if(this[e]===t)return this;for(let n=0,r=this.children.length;n<r;n++){const a=this.children[n].getObjectByProperty(e,t);if(a!==void 0)return a}}getObjectsByProperty(e,t,n=[]){this[e]===t&&n.push(this);const r=this.children;for(let s=0,a=r.length;s<a;s++)r[s].getObjectsByProperty(e,t,n);return n}getWorldPosition(e){return this.updateWorldMatrix(!0,!1),e.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(ar,e,ju),e}getWorldScale(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(ar,Zu,e),e}getWorldDirection(e){this.updateWorldMatrix(!0,!1);const t=this.matrixWorld.elements;return e.set(t[8],t[9],t[10]).normalize()}raycast(){}traverse(e){e(this);const t=this.children;for(let n=0,r=t.length;n<r;n++)t[n].traverse(e)}traverseVisible(e){if(this.visible===!1)return;e(this);const t=this.children;for(let n=0,r=t.length;n<r;n++)t[n].traverseVisible(e)}traverseAncestors(e){const t=this.parent;t!==null&&(e(t),t.traverseAncestors(e))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale),this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(e){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||e)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,e=!0);const t=this.children;for(let n=0,r=t.length;n<r;n++)t[n].updateMatrixWorld(e)}updateWorldMatrix(e,t){const n=this.parent;if(e===!0&&n!==null&&n.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),t===!0){const r=this.children;for(let s=0,a=r.length;s<a;s++)r[s].updateWorldMatrix(!1,!0)}}toJSON(e){const t=e===void 0||typeof e=="string",n={};t&&(e={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},n.metadata={version:4.7,type:"Object",generator:"Object3D.toJSON"});const r={};r.uuid=this.uuid,r.type=this.type,this.name!==""&&(r.name=this.name),this.castShadow===!0&&(r.castShadow=!0),this.receiveShadow===!0&&(r.receiveShadow=!0),this.visible===!1&&(r.visible=!1),this.frustumCulled===!1&&(r.frustumCulled=!1),this.renderOrder!==0&&(r.renderOrder=this.renderOrder),Object.keys(this.userData).length>0&&(r.userData=this.userData),r.layers=this.layers.mask,r.matrix=this.matrix.toArray(),r.up=this.up.toArray(),this.matrixAutoUpdate===!1&&(r.matrixAutoUpdate=!1),this.isInstancedMesh&&(r.type="InstancedMesh",r.count=this.count,r.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(r.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(r.type="BatchedMesh",r.perObjectFrustumCulled=this.perObjectFrustumCulled,r.sortObjects=this.sortObjects,r.drawRanges=this._drawRanges,r.reservedRanges=this._reservedRanges,r.geometryInfo=this._geometryInfo.map(o=>({...o,boundingBox:o.boundingBox?o.boundingBox.toJSON():void 0,boundingSphere:o.boundingSphere?o.boundingSphere.toJSON():void 0})),r.instanceInfo=this._instanceInfo.map(o=>({...o})),r.availableInstanceIds=this._availableInstanceIds.slice(),r.availableGeometryIds=this._availableGeometryIds.slice(),r.nextIndexStart=this._nextIndexStart,r.nextVertexStart=this._nextVertexStart,r.geometryCount=this._geometryCount,r.maxInstanceCount=this._maxInstanceCount,r.maxVertexCount=this._maxVertexCount,r.maxIndexCount=this._maxIndexCount,r.geometryInitialized=this._geometryInitialized,r.matricesTexture=this._matricesTexture.toJSON(e),r.indirectTexture=this._indirectTexture.toJSON(e),this._colorsTexture!==null&&(r.colorsTexture=this._colorsTexture.toJSON(e)),this.boundingSphere!==null&&(r.boundingSphere=this.boundingSphere.toJSON()),this.boundingBox!==null&&(r.boundingBox=this.boundingBox.toJSON()));function s(o,u){return o[u.uuid]===void 0&&(o[u.uuid]=u.toJSON(e)),u.uuid}if(this.isScene)this.background&&(this.background.isColor?r.background=this.background.toJSON():this.background.isTexture&&(r.background=this.background.toJSON(e).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(r.environment=this.environment.toJSON(e).uuid);else if(this.isMesh||this.isLine||this.isPoints){r.geometry=s(e.geometries,this.geometry);const o=this.geometry.parameters;if(o!==void 0&&o.shapes!==void 0){const u=o.shapes;if(Array.isArray(u))for(let l=0,h=u.length;l<h;l++){const p=u[l];s(e.shapes,p)}else s(e.shapes,u)}}if(this.isSkinnedMesh&&(r.bindMode=this.bindMode,r.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(s(e.skeletons,this.skeleton),r.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){const o=[];for(let u=0,l=this.material.length;u<l;u++)o.push(s(e.materials,this.material[u]));r.material=o}else r.material=s(e.materials,this.material);if(this.children.length>0){r.children=[];for(let o=0;o<this.children.length;o++)r.children.push(this.children[o].toJSON(e).object)}if(this.animations.length>0){r.animations=[];for(let o=0;o<this.animations.length;o++){const u=this.animations[o];r.animations.push(s(e.animations,u))}}if(t){const o=a(e.geometries),u=a(e.materials),l=a(e.textures),h=a(e.images),p=a(e.shapes),m=a(e.skeletons),v=a(e.animations),y=a(e.nodes);o.length>0&&(n.geometries=o),u.length>0&&(n.materials=u),l.length>0&&(n.textures=l),h.length>0&&(n.images=h),p.length>0&&(n.shapes=p),m.length>0&&(n.skeletons=m),v.length>0&&(n.animations=v),y.length>0&&(n.nodes=y)}return n.object=r,n;function a(o){const u=[];for(const l in o){const h=o[l];delete h.metadata,u.push(h)}return u}}clone(e){return new this.constructor().copy(this,e)}copy(e,t=!0){if(this.name=e.name,this.up.copy(e.up),this.position.copy(e.position),this.rotation.order=e.rotation.order,this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.matrix.copy(e.matrix),this.matrixWorld.copy(e.matrixWorld),this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrixWorldAutoUpdate=e.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=e.matrixWorldNeedsUpdate,this.layers.mask=e.layers.mask,this.visible=e.visible,this.castShadow=e.castShadow,this.receiveShadow=e.receiveShadow,this.frustumCulled=e.frustumCulled,this.renderOrder=e.renderOrder,this.animations=e.animations.slice(),this.userData=JSON.parse(JSON.stringify(e.userData)),t===!0)for(let n=0;n<e.children.length;n++){const r=e.children[n];this.add(r.clone())}return this}}ln.DEFAULT_UP=new $(0,1,0);ln.DEFAULT_MATRIX_AUTO_UPDATE=!0;ln.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;const Sn=new $,Hn=new $,Us=new $,kn=new $,Ii=new $,Fi=new $,Go=new $,Is=new $,Fs=new $,Ns=new $,Os=new Nt,Bs=new Nt,Vs=new Nt;class yn{constructor(e=new $,t=new $,n=new $){this.a=e,this.b=t,this.c=n}static getNormal(e,t,n,r){r.subVectors(n,t),Sn.subVectors(e,t),r.cross(Sn);const s=r.lengthSq();return s>0?r.multiplyScalar(1/Math.sqrt(s)):r.set(0,0,0)}static getBarycoord(e,t,n,r,s){Sn.subVectors(r,t),Hn.subVectors(n,t),Us.subVectors(e,t);const a=Sn.dot(Sn),o=Sn.dot(Hn),u=Sn.dot(Us),l=Hn.dot(Hn),h=Hn.dot(Us),p=a*l-o*o;if(p===0)return s.set(0,0,0),null;const m=1/p,v=(l*u-o*h)*m,y=(a*h-o*u)*m;return s.set(1-v-y,y,v)}static containsPoint(e,t,n,r){return this.getBarycoord(e,t,n,r,kn)===null?!1:kn.x>=0&&kn.y>=0&&kn.x+kn.y<=1}static getInterpolation(e,t,n,r,s,a,o,u){return this.getBarycoord(e,t,n,r,kn)===null?(u.x=0,u.y=0,"z"in u&&(u.z=0),"w"in u&&(u.w=0),null):(u.setScalar(0),u.addScaledVector(s,kn.x),u.addScaledVector(a,kn.y),u.addScaledVector(o,kn.z),u)}static getInterpolatedAttribute(e,t,n,r,s,a){return Os.setScalar(0),Bs.setScalar(0),Vs.setScalar(0),Os.fromBufferAttribute(e,t),Bs.fromBufferAttribute(e,n),Vs.fromBufferAttribute(e,r),a.setScalar(0),a.addScaledVector(Os,s.x),a.addScaledVector(Bs,s.y),a.addScaledVector(Vs,s.z),a}static isFrontFacing(e,t,n,r){return Sn.subVectors(n,t),Hn.subVectors(e,t),Sn.cross(Hn).dot(r)<0}set(e,t,n){return this.a.copy(e),this.b.copy(t),this.c.copy(n),this}setFromPointsAndIndices(e,t,n,r){return this.a.copy(e[t]),this.b.copy(e[n]),this.c.copy(e[r]),this}setFromAttributeAndIndices(e,t,n,r){return this.a.fromBufferAttribute(e,t),this.b.fromBufferAttribute(e,n),this.c.fromBufferAttribute(e,r),this}clone(){return new this.constructor().copy(this)}copy(e){return this.a.copy(e.a),this.b.copy(e.b),this.c.copy(e.c),this}getArea(){return Sn.subVectors(this.c,this.b),Hn.subVectors(this.a,this.b),Sn.cross(Hn).length()*.5}getMidpoint(e){return e.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(e){return yn.getNormal(this.a,this.b,this.c,e)}getPlane(e){return e.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(e,t){return yn.getBarycoord(e,this.a,this.b,this.c,t)}getInterpolation(e,t,n,r,s){return yn.getInterpolation(e,this.a,this.b,this.c,t,n,r,s)}containsPoint(e){return yn.containsPoint(e,this.a,this.b,this.c)}isFrontFacing(e){return yn.isFrontFacing(this.a,this.b,this.c,e)}intersectsBox(e){return e.intersectsTriangle(this)}closestPointToPoint(e,t){const n=this.a,r=this.b,s=this.c;let a,o;Ii.subVectors(r,n),Fi.subVectors(s,n),Is.subVectors(e,n);const u=Ii.dot(Is),l=Fi.dot(Is);if(u<=0&&l<=0)return t.copy(n);Fs.subVectors(e,r);const h=Ii.dot(Fs),p=Fi.dot(Fs);if(h>=0&&p<=h)return t.copy(r);const m=u*p-h*l;if(m<=0&&u>=0&&h<=0)return a=u/(u-h),t.copy(n).addScaledVector(Ii,a);Ns.subVectors(e,s);const v=Ii.dot(Ns),y=Fi.dot(Ns);if(y>=0&&v<=y)return t.copy(s);const b=v*l-u*y;if(b<=0&&l>=0&&y<=0)return o=l/(l-y),t.copy(n).addScaledVector(Fi,o);const g=h*y-v*p;if(g<=0&&p-h>=0&&v-y>=0)return Go.subVectors(s,r),o=(p-h)/(p-h+(v-y)),t.copy(r).addScaledVector(Go,o);const d=1/(g+b+m);return a=b*d,o=m*d,t.copy(n).addScaledVector(Ii,a).addScaledVector(Fi,o)}equals(e){return e.a.equals(this.a)&&e.b.equals(this.b)&&e.c.equals(this.c)}}const zl={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},ni={h:0,s:0,l:0},Or={h:0,s:0,l:0};function zs(i,e,t){return t<0&&(t+=1),t>1&&(t-=1),t<1/6?i+(e-i)*6*t:t<1/2?e:t<2/3?i+(e-i)*6*(2/3-t):i}class yt{constructor(e,t,n){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(e,t,n)}set(e,t,n){if(t===void 0&&n===void 0){const r=e;r&&r.isColor?this.copy(r):typeof r=="number"?this.setHex(r):typeof r=="string"&&this.setStyle(r)}else this.setRGB(e,t,n);return this}setScalar(e){return this.r=e,this.g=e,this.b=e,this}setHex(e,t=mn){return e=Math.floor(e),this.r=(e>>16&255)/255,this.g=(e>>8&255)/255,this.b=(e&255)/255,_t.colorSpaceToWorking(this,t),this}setRGB(e,t,n,r=_t.workingColorSpace){return this.r=e,this.g=t,this.b=n,_t.colorSpaceToWorking(this,r),this}setHSL(e,t,n,r=_t.workingColorSpace){if(e=Bu(e,1),t=dt(t,0,1),n=dt(n,0,1),t===0)this.r=this.g=this.b=n;else{const s=n<=.5?n*(1+t):n+t-n*t,a=2*n-s;this.r=zs(a,s,e+1/3),this.g=zs(a,s,e),this.b=zs(a,s,e-1/3)}return _t.colorSpaceToWorking(this,r),this}setStyle(e,t=mn){function n(s){s!==void 0&&parseFloat(s)<1&&et("Color: Alpha component of "+e+" will be ignored.")}let r;if(r=/^(\w+)\(([^\)]*)\)/.exec(e)){let s;const a=r[1],o=r[2];switch(a){case"rgb":case"rgba":if(s=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return n(s[4]),this.setRGB(Math.min(255,parseInt(s[1],10))/255,Math.min(255,parseInt(s[2],10))/255,Math.min(255,parseInt(s[3],10))/255,t);if(s=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return n(s[4]),this.setRGB(Math.min(100,parseInt(s[1],10))/100,Math.min(100,parseInt(s[2],10))/100,Math.min(100,parseInt(s[3],10))/100,t);break;case"hsl":case"hsla":if(s=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return n(s[4]),this.setHSL(parseFloat(s[1])/360,parseFloat(s[2])/100,parseFloat(s[3])/100,t);break;default:et("Color: Unknown color model "+e)}}else if(r=/^\#([A-Fa-f\d]+)$/.exec(e)){const s=r[1],a=s.length;if(a===3)return this.setRGB(parseInt(s.charAt(0),16)/15,parseInt(s.charAt(1),16)/15,parseInt(s.charAt(2),16)/15,t);if(a===6)return this.setHex(parseInt(s,16),t);et("Color: Invalid hex color "+e)}else if(e&&e.length>0)return this.setColorName(e,t);return this}setColorName(e,t=mn){const n=zl[e.toLowerCase()];return n!==void 0?this.setHex(n,t):et("Color: Unknown color "+e),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(e){return this.r=e.r,this.g=e.g,this.b=e.b,this}copySRGBToLinear(e){return this.r=qn(e.r),this.g=qn(e.g),this.b=qn(e.b),this}copyLinearToSRGB(e){return this.r=Gi(e.r),this.g=Gi(e.g),this.b=Gi(e.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(e=mn){return _t.workingToColorSpace(Jt.copy(this),e),Math.round(dt(Jt.r*255,0,255))*65536+Math.round(dt(Jt.g*255,0,255))*256+Math.round(dt(Jt.b*255,0,255))}getHexString(e=mn){return("000000"+this.getHex(e).toString(16)).slice(-6)}getHSL(e,t=_t.workingColorSpace){_t.workingToColorSpace(Jt.copy(this),t);const n=Jt.r,r=Jt.g,s=Jt.b,a=Math.max(n,r,s),o=Math.min(n,r,s);let u,l;const h=(o+a)/2;if(o===a)u=0,l=0;else{const p=a-o;switch(l=h<=.5?p/(a+o):p/(2-a-o),a){case n:u=(r-s)/p+(r<s?6:0);break;case r:u=(s-n)/p+2;break;case s:u=(n-r)/p+4;break}u/=6}return e.h=u,e.s=l,e.l=h,e}getRGB(e,t=_t.workingColorSpace){return _t.workingToColorSpace(Jt.copy(this),t),e.r=Jt.r,e.g=Jt.g,e.b=Jt.b,e}getStyle(e=mn){_t.workingToColorSpace(Jt.copy(this),e);const t=Jt.r,n=Jt.g,r=Jt.b;return e!==mn?`color(${e} ${t.toFixed(3)} ${n.toFixed(3)} ${r.toFixed(3)})`:`rgb(${Math.round(t*255)},${Math.round(n*255)},${Math.round(r*255)})`}offsetHSL(e,t,n){return this.getHSL(ni),this.setHSL(ni.h+e,ni.s+t,ni.l+n)}add(e){return this.r+=e.r,this.g+=e.g,this.b+=e.b,this}addColors(e,t){return this.r=e.r+t.r,this.g=e.g+t.g,this.b=e.b+t.b,this}addScalar(e){return this.r+=e,this.g+=e,this.b+=e,this}sub(e){return this.r=Math.max(0,this.r-e.r),this.g=Math.max(0,this.g-e.g),this.b=Math.max(0,this.b-e.b),this}multiply(e){return this.r*=e.r,this.g*=e.g,this.b*=e.b,this}multiplyScalar(e){return this.r*=e,this.g*=e,this.b*=e,this}lerp(e,t){return this.r+=(e.r-this.r)*t,this.g+=(e.g-this.g)*t,this.b+=(e.b-this.b)*t,this}lerpColors(e,t,n){return this.r=e.r+(t.r-e.r)*n,this.g=e.g+(t.g-e.g)*n,this.b=e.b+(t.b-e.b)*n,this}lerpHSL(e,t){this.getHSL(ni),e.getHSL(Or);const n=ys(ni.h,Or.h,t),r=ys(ni.s,Or.s,t),s=ys(ni.l,Or.l,t);return this.setHSL(n,r,s),this}setFromVector3(e){return this.r=e.x,this.g=e.y,this.b=e.z,this}applyMatrix3(e){const t=this.r,n=this.g,r=this.b,s=e.elements;return this.r=s[0]*t+s[3]*n+s[6]*r,this.g=s[1]*t+s[4]*n+s[7]*r,this.b=s[2]*t+s[5]*n+s[8]*r,this}equals(e){return e.r===this.r&&e.g===this.g&&e.b===this.b}fromArray(e,t=0){return this.r=e[t],this.g=e[t+1],this.b=e[t+2],this}toArray(e=[],t=0){return e[t]=this.r,e[t+1]=this.g,e[t+2]=this.b,e}fromBufferAttribute(e,t){return this.r=e.getX(t),this.g=e.getY(t),this.b=e.getZ(t),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}}const Jt=new yt;yt.NAMES=zl;let Qu=0;class Yi extends qi{constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:Qu++}),this.uuid=gr(),this.name="",this.type="Material",this.blending=zi,this.side=ai,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=Js,this.blendDst=Qs,this.blendEquation=gi,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new yt(0,0,0),this.blendAlpha=0,this.depthFunc=Hi,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=Ao,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=Ai,this.stencilZFail=Ai,this.stencilZPass=Ai,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.allowOverride=!0,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(e){this._alphaTest>0!=e>0&&this.version++,this._alphaTest=e}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(e){if(e!==void 0)for(const t in e){const n=e[t];if(n===void 0){et(`Material: parameter '${t}' has value of undefined.`);continue}const r=this[t];if(r===void 0){et(`Material: '${t}' is not a property of THREE.${this.type}.`);continue}r&&r.isColor?r.set(n):r&&r.isVector3&&n&&n.isVector3?r.copy(n):this[t]=n}}toJSON(e){const t=e===void 0||typeof e=="string";t&&(e={textures:{},images:{}});const n={metadata:{version:4.7,type:"Material",generator:"Material.toJSON"}};n.uuid=this.uuid,n.type=this.type,this.name!==""&&(n.name=this.name),this.color&&this.color.isColor&&(n.color=this.color.getHex()),this.roughness!==void 0&&(n.roughness=this.roughness),this.metalness!==void 0&&(n.metalness=this.metalness),this.sheen!==void 0&&(n.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(n.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(n.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(n.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&this.emissiveIntensity!==1&&(n.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(n.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(n.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(n.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(n.shininess=this.shininess),this.clearcoat!==void 0&&(n.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(n.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(n.clearcoatMap=this.clearcoatMap.toJSON(e).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(n.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(e).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(n.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(e).uuid,n.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.sheenColorMap&&this.sheenColorMap.isTexture&&(n.sheenColorMap=this.sheenColorMap.toJSON(e).uuid),this.sheenRoughnessMap&&this.sheenRoughnessMap.isTexture&&(n.sheenRoughnessMap=this.sheenRoughnessMap.toJSON(e).uuid),this.dispersion!==void 0&&(n.dispersion=this.dispersion),this.iridescence!==void 0&&(n.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(n.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(n.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(n.iridescenceMap=this.iridescenceMap.toJSON(e).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(n.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(e).uuid),this.anisotropy!==void 0&&(n.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(n.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(n.anisotropyMap=this.anisotropyMap.toJSON(e).uuid),this.map&&this.map.isTexture&&(n.map=this.map.toJSON(e).uuid),this.matcap&&this.matcap.isTexture&&(n.matcap=this.matcap.toJSON(e).uuid),this.alphaMap&&this.alphaMap.isTexture&&(n.alphaMap=this.alphaMap.toJSON(e).uuid),this.lightMap&&this.lightMap.isTexture&&(n.lightMap=this.lightMap.toJSON(e).uuid,n.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(n.aoMap=this.aoMap.toJSON(e).uuid,n.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(n.bumpMap=this.bumpMap.toJSON(e).uuid,n.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(n.normalMap=this.normalMap.toJSON(e).uuid,n.normalMapType=this.normalMapType,n.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(n.displacementMap=this.displacementMap.toJSON(e).uuid,n.displacementScale=this.displacementScale,n.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(n.roughnessMap=this.roughnessMap.toJSON(e).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(n.metalnessMap=this.metalnessMap.toJSON(e).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(n.emissiveMap=this.emissiveMap.toJSON(e).uuid),this.specularMap&&this.specularMap.isTexture&&(n.specularMap=this.specularMap.toJSON(e).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(n.specularIntensityMap=this.specularIntensityMap.toJSON(e).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(n.specularColorMap=this.specularColorMap.toJSON(e).uuid),this.envMap&&this.envMap.isTexture&&(n.envMap=this.envMap.toJSON(e).uuid,this.combine!==void 0&&(n.combine=this.combine)),this.envMapRotation!==void 0&&(n.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(n.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(n.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(n.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(n.gradientMap=this.gradientMap.toJSON(e).uuid),this.transmission!==void 0&&(n.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(n.transmissionMap=this.transmissionMap.toJSON(e).uuid),this.thickness!==void 0&&(n.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(n.thicknessMap=this.thicknessMap.toJSON(e).uuid),this.attenuationDistance!==void 0&&this.attenuationDistance!==1/0&&(n.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(n.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(n.size=this.size),this.shadowSide!==null&&(n.shadowSide=this.shadowSide),this.sizeAttenuation!==void 0&&(n.sizeAttenuation=this.sizeAttenuation),this.blending!==zi&&(n.blending=this.blending),this.side!==ai&&(n.side=this.side),this.vertexColors===!0&&(n.vertexColors=!0),this.opacity<1&&(n.opacity=this.opacity),this.transparent===!0&&(n.transparent=!0),this.blendSrc!==Js&&(n.blendSrc=this.blendSrc),this.blendDst!==Qs&&(n.blendDst=this.blendDst),this.blendEquation!==gi&&(n.blendEquation=this.blendEquation),this.blendSrcAlpha!==null&&(n.blendSrcAlpha=this.blendSrcAlpha),this.blendDstAlpha!==null&&(n.blendDstAlpha=this.blendDstAlpha),this.blendEquationAlpha!==null&&(n.blendEquationAlpha=this.blendEquationAlpha),this.blendColor&&this.blendColor.isColor&&(n.blendColor=this.blendColor.getHex()),this.blendAlpha!==0&&(n.blendAlpha=this.blendAlpha),this.depthFunc!==Hi&&(n.depthFunc=this.depthFunc),this.depthTest===!1&&(n.depthTest=this.depthTest),this.depthWrite===!1&&(n.depthWrite=this.depthWrite),this.colorWrite===!1&&(n.colorWrite=this.colorWrite),this.stencilWriteMask!==255&&(n.stencilWriteMask=this.stencilWriteMask),this.stencilFunc!==Ao&&(n.stencilFunc=this.stencilFunc),this.stencilRef!==0&&(n.stencilRef=this.stencilRef),this.stencilFuncMask!==255&&(n.stencilFuncMask=this.stencilFuncMask),this.stencilFail!==Ai&&(n.stencilFail=this.stencilFail),this.stencilZFail!==Ai&&(n.stencilZFail=this.stencilZFail),this.stencilZPass!==Ai&&(n.stencilZPass=this.stencilZPass),this.stencilWrite===!0&&(n.stencilWrite=this.stencilWrite),this.rotation!==void 0&&this.rotation!==0&&(n.rotation=this.rotation),this.polygonOffset===!0&&(n.polygonOffset=!0),this.polygonOffsetFactor!==0&&(n.polygonOffsetFactor=this.polygonOffsetFactor),this.polygonOffsetUnits!==0&&(n.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth!==void 0&&this.linewidth!==1&&(n.linewidth=this.linewidth),this.dashSize!==void 0&&(n.dashSize=this.dashSize),this.gapSize!==void 0&&(n.gapSize=this.gapSize),this.scale!==void 0&&(n.scale=this.scale),this.dithering===!0&&(n.dithering=!0),this.alphaTest>0&&(n.alphaTest=this.alphaTest),this.alphaHash===!0&&(n.alphaHash=!0),this.alphaToCoverage===!0&&(n.alphaToCoverage=!0),this.premultipliedAlpha===!0&&(n.premultipliedAlpha=!0),this.forceSinglePass===!0&&(n.forceSinglePass=!0),this.allowOverride===!1&&(n.allowOverride=!1),this.wireframe===!0&&(n.wireframe=!0),this.wireframeLinewidth>1&&(n.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!=="round"&&(n.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!=="round"&&(n.wireframeLinejoin=this.wireframeLinejoin),this.flatShading===!0&&(n.flatShading=!0),this.visible===!1&&(n.visible=!1),this.toneMapped===!1&&(n.toneMapped=!1),this.fog===!1&&(n.fog=!1),Object.keys(this.userData).length>0&&(n.userData=this.userData);function r(s){const a=[];for(const o in s){const u=s[o];delete u.metadata,a.push(u)}return a}if(t){const s=r(e.textures),a=r(e.images);s.length>0&&(n.textures=s),a.length>0&&(n.images=a)}return n}clone(){return new this.constructor().copy(this)}copy(e){this.name=e.name,this.blending=e.blending,this.side=e.side,this.vertexColors=e.vertexColors,this.opacity=e.opacity,this.transparent=e.transparent,this.blendSrc=e.blendSrc,this.blendDst=e.blendDst,this.blendEquation=e.blendEquation,this.blendSrcAlpha=e.blendSrcAlpha,this.blendDstAlpha=e.blendDstAlpha,this.blendEquationAlpha=e.blendEquationAlpha,this.blendColor.copy(e.blendColor),this.blendAlpha=e.blendAlpha,this.depthFunc=e.depthFunc,this.depthTest=e.depthTest,this.depthWrite=e.depthWrite,this.stencilWriteMask=e.stencilWriteMask,this.stencilFunc=e.stencilFunc,this.stencilRef=e.stencilRef,this.stencilFuncMask=e.stencilFuncMask,this.stencilFail=e.stencilFail,this.stencilZFail=e.stencilZFail,this.stencilZPass=e.stencilZPass,this.stencilWrite=e.stencilWrite;const t=e.clippingPlanes;let n=null;if(t!==null){const r=t.length;n=new Array(r);for(let s=0;s!==r;++s)n[s]=t[s].clone()}return this.clippingPlanes=n,this.clipIntersection=e.clipIntersection,this.clipShadows=e.clipShadows,this.shadowSide=e.shadowSide,this.colorWrite=e.colorWrite,this.precision=e.precision,this.polygonOffset=e.polygonOffset,this.polygonOffsetFactor=e.polygonOffsetFactor,this.polygonOffsetUnits=e.polygonOffsetUnits,this.dithering=e.dithering,this.alphaTest=e.alphaTest,this.alphaHash=e.alphaHash,this.alphaToCoverage=e.alphaToCoverage,this.premultipliedAlpha=e.premultipliedAlpha,this.forceSinglePass=e.forceSinglePass,this.allowOverride=e.allowOverride,this.visible=e.visible,this.toneMapped=e.toneMapped,this.userData=JSON.parse(JSON.stringify(e.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(e){e===!0&&this.version++}}class Gl extends Yi{constructor(e){super(),this.isMeshBasicMaterial=!0,this.type="MeshBasicMaterial",this.color=new yt(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new Un,this.combine=Ya,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.fog=e.fog,this}}const zt=new $,Br=new pt;let ef=0;class _n{constructor(e,t,n=!1){if(Array.isArray(e))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,Object.defineProperty(this,"id",{value:ef++}),this.name="",this.array=e,this.itemSize=t,this.count=e!==void 0?e.length/t:0,this.normalized=n,this.usage=wo,this.updateRanges=[],this.gpuType=Rn,this.version=0}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.name=e.name,this.array=new e.array.constructor(e.array),this.itemSize=e.itemSize,this.count=e.count,this.normalized=e.normalized,this.usage=e.usage,this.gpuType=e.gpuType,this}copyAt(e,t,n){e*=this.itemSize,n*=t.itemSize;for(let r=0,s=this.itemSize;r<s;r++)this.array[e+r]=t.array[n+r];return this}copyArray(e){return this.array.set(e),this}applyMatrix3(e){if(this.itemSize===2)for(let t=0,n=this.count;t<n;t++)Br.fromBufferAttribute(this,t),Br.applyMatrix3(e),this.setXY(t,Br.x,Br.y);else if(this.itemSize===3)for(let t=0,n=this.count;t<n;t++)zt.fromBufferAttribute(this,t),zt.applyMatrix3(e),this.setXYZ(t,zt.x,zt.y,zt.z);return this}applyMatrix4(e){for(let t=0,n=this.count;t<n;t++)zt.fromBufferAttribute(this,t),zt.applyMatrix4(e),this.setXYZ(t,zt.x,zt.y,zt.z);return this}applyNormalMatrix(e){for(let t=0,n=this.count;t<n;t++)zt.fromBufferAttribute(this,t),zt.applyNormalMatrix(e),this.setXYZ(t,zt.x,zt.y,zt.z);return this}transformDirection(e){for(let t=0,n=this.count;t<n;t++)zt.fromBufferAttribute(this,t),zt.transformDirection(e),this.setXYZ(t,zt.x,zt.y,zt.z);return this}set(e,t=0){return this.array.set(e,t),this}getComponent(e,t){let n=this.array[e*this.itemSize+t];return this.normalized&&(n=ir(n,this.array)),n}setComponent(e,t,n){return this.normalized&&(n=an(n,this.array)),this.array[e*this.itemSize+t]=n,this}getX(e){let t=this.array[e*this.itemSize];return this.normalized&&(t=ir(t,this.array)),t}setX(e,t){return this.normalized&&(t=an(t,this.array)),this.array[e*this.itemSize]=t,this}getY(e){let t=this.array[e*this.itemSize+1];return this.normalized&&(t=ir(t,this.array)),t}setY(e,t){return this.normalized&&(t=an(t,this.array)),this.array[e*this.itemSize+1]=t,this}getZ(e){let t=this.array[e*this.itemSize+2];return this.normalized&&(t=ir(t,this.array)),t}setZ(e,t){return this.normalized&&(t=an(t,this.array)),this.array[e*this.itemSize+2]=t,this}getW(e){let t=this.array[e*this.itemSize+3];return this.normalized&&(t=ir(t,this.array)),t}setW(e,t){return this.normalized&&(t=an(t,this.array)),this.array[e*this.itemSize+3]=t,this}setXY(e,t,n){return e*=this.itemSize,this.normalized&&(t=an(t,this.array),n=an(n,this.array)),this.array[e+0]=t,this.array[e+1]=n,this}setXYZ(e,t,n,r){return e*=this.itemSize,this.normalized&&(t=an(t,this.array),n=an(n,this.array),r=an(r,this.array)),this.array[e+0]=t,this.array[e+1]=n,this.array[e+2]=r,this}setXYZW(e,t,n,r,s){return e*=this.itemSize,this.normalized&&(t=an(t,this.array),n=an(n,this.array),r=an(r,this.array),s=an(s,this.array)),this.array[e+0]=t,this.array[e+1]=n,this.array[e+2]=r,this.array[e+3]=s,this}onUpload(e){return this.onUploadCallback=e,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){const e={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return this.name!==""&&(e.name=this.name),this.usage!==wo&&(e.usage=this.usage),e}}class Hl extends _n{constructor(e,t,n){super(new Uint16Array(e),t,n)}}class kl extends _n{constructor(e,t,n){super(new Uint32Array(e),t,n)}}class gn extends _n{constructor(e,t,n){super(new Float32Array(e),t,n)}}let tf=0;const pn=new Ot,Gs=new ln,Ni=new $,fn=new xr,or=new xr,kt=new $;class bn extends qi{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:tf++}),this.uuid=gr(),this.name="",this.type="BufferGeometry",this.index=null,this.indirect=null,this.indirectOffset=0,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={}}getIndex(){return this.index}setIndex(e){return Array.isArray(e)?this.index=new(Ol(e)?kl:Hl)(e,1):this.index=e,this}setIndirect(e,t=0){return this.indirect=e,this.indirectOffset=t,this}getIndirect(){return this.indirect}getAttribute(e){return this.attributes[e]}setAttribute(e,t){return this.attributes[e]=t,this}deleteAttribute(e){return delete this.attributes[e],this}hasAttribute(e){return this.attributes[e]!==void 0}addGroup(e,t,n=0){this.groups.push({start:e,count:t,materialIndex:n})}clearGroups(){this.groups=[]}setDrawRange(e,t){this.drawRange.start=e,this.drawRange.count=t}applyMatrix4(e){const t=this.attributes.position;t!==void 0&&(t.applyMatrix4(e),t.needsUpdate=!0);const n=this.attributes.normal;if(n!==void 0){const s=new st().getNormalMatrix(e);n.applyNormalMatrix(s),n.needsUpdate=!0}const r=this.attributes.tangent;return r!==void 0&&(r.transformDirection(e),r.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this}applyQuaternion(e){return pn.makeRotationFromQuaternion(e),this.applyMatrix4(pn),this}rotateX(e){return pn.makeRotationX(e),this.applyMatrix4(pn),this}rotateY(e){return pn.makeRotationY(e),this.applyMatrix4(pn),this}rotateZ(e){return pn.makeRotationZ(e),this.applyMatrix4(pn),this}translate(e,t,n){return pn.makeTranslation(e,t,n),this.applyMatrix4(pn),this}scale(e,t,n){return pn.makeScale(e,t,n),this.applyMatrix4(pn),this}lookAt(e){return Gs.lookAt(e),Gs.updateMatrix(),this.applyMatrix4(Gs.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(Ni).negate(),this.translate(Ni.x,Ni.y,Ni.z),this}setFromPoints(e){const t=this.getAttribute("position");if(t===void 0){const n=[];for(let r=0,s=e.length;r<s;r++){const a=e[r];n.push(a.x,a.y,a.z||0)}this.setAttribute("position",new gn(n,3))}else{const n=Math.min(e.length,t.count);for(let r=0;r<n;r++){const s=e[r];t.setXYZ(r,s.x,s.y,s.z||0)}e.length>t.count&&et("BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."),t.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new xr);const e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){xt("BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.",this),this.boundingBox.set(new $(-1/0,-1/0,-1/0),new $(1/0,1/0,1/0));return}if(e!==void 0){if(this.boundingBox.setFromBufferAttribute(e),t)for(let n=0,r=t.length;n<r;n++){const s=t[n];fn.setFromBufferAttribute(s),this.morphTargetsRelative?(kt.addVectors(this.boundingBox.min,fn.min),this.boundingBox.expandByPoint(kt),kt.addVectors(this.boundingBox.max,fn.max),this.boundingBox.expandByPoint(kt)):(this.boundingBox.expandByPoint(fn.min),this.boundingBox.expandByPoint(fn.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&xt('BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new so);const e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){xt("BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.",this),this.boundingSphere.set(new $,1/0);return}if(e){const n=this.boundingSphere.center;if(fn.setFromBufferAttribute(e),t)for(let s=0,a=t.length;s<a;s++){const o=t[s];or.setFromBufferAttribute(o),this.morphTargetsRelative?(kt.addVectors(fn.min,or.min),fn.expandByPoint(kt),kt.addVectors(fn.max,or.max),fn.expandByPoint(kt)):(fn.expandByPoint(or.min),fn.expandByPoint(or.max))}fn.getCenter(n);let r=0;for(let s=0,a=e.count;s<a;s++)kt.fromBufferAttribute(e,s),r=Math.max(r,n.distanceToSquared(kt));if(t)for(let s=0,a=t.length;s<a;s++){const o=t[s],u=this.morphTargetsRelative;for(let l=0,h=o.count;l<h;l++)kt.fromBufferAttribute(o,l),u&&(Ni.fromBufferAttribute(e,l),kt.add(Ni)),r=Math.max(r,n.distanceToSquared(kt))}this.boundingSphere.radius=Math.sqrt(r),isNaN(this.boundingSphere.radius)&&xt('BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){const e=this.index,t=this.attributes;if(e===null||t.position===void 0||t.normal===void 0||t.uv===void 0){xt("BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}const n=t.position,r=t.normal,s=t.uv;this.hasAttribute("tangent")===!1&&this.setAttribute("tangent",new _n(new Float32Array(4*n.count),4));const a=this.getAttribute("tangent"),o=[],u=[];for(let G=0;G<n.count;G++)o[G]=new $,u[G]=new $;const l=new $,h=new $,p=new $,m=new pt,v=new pt,y=new pt,b=new $,g=new $;function d(G,x,E){l.fromBufferAttribute(n,G),h.fromBufferAttribute(n,x),p.fromBufferAttribute(n,E),m.fromBufferAttribute(s,G),v.fromBufferAttribute(s,x),y.fromBufferAttribute(s,E),h.sub(l),p.sub(l),v.sub(m),y.sub(m);const B=1/(v.x*y.y-y.x*v.y);isFinite(B)&&(b.copy(h).multiplyScalar(y.y).addScaledVector(p,-v.y).multiplyScalar(B),g.copy(p).multiplyScalar(v.x).addScaledVector(h,-y.x).multiplyScalar(B),o[G].add(b),o[x].add(b),o[E].add(b),u[G].add(g),u[x].add(g),u[E].add(g))}let R=this.groups;R.length===0&&(R=[{start:0,count:e.count}]);for(let G=0,x=R.length;G<x;++G){const E=R[G],B=E.start,j=E.count;for(let K=B,ie=B+j;K<ie;K+=3)d(e.getX(K+0),e.getX(K+1),e.getX(K+2))}const C=new $,A=new $,P=new $,D=new $;function I(G){P.fromBufferAttribute(r,G),D.copy(P);const x=o[G];C.copy(x),C.sub(P.multiplyScalar(P.dot(x))).normalize(),A.crossVectors(D,x);const B=A.dot(u[G])<0?-1:1;a.setXYZW(G,C.x,C.y,C.z,B)}for(let G=0,x=R.length;G<x;++G){const E=R[G],B=E.start,j=E.count;for(let K=B,ie=B+j;K<ie;K+=3)I(e.getX(K+0)),I(e.getX(K+1)),I(e.getX(K+2))}}computeVertexNormals(){const e=this.index,t=this.getAttribute("position");if(t!==void 0){let n=this.getAttribute("normal");if(n===void 0)n=new _n(new Float32Array(t.count*3),3),this.setAttribute("normal",n);else for(let m=0,v=n.count;m<v;m++)n.setXYZ(m,0,0,0);const r=new $,s=new $,a=new $,o=new $,u=new $,l=new $,h=new $,p=new $;if(e)for(let m=0,v=e.count;m<v;m+=3){const y=e.getX(m+0),b=e.getX(m+1),g=e.getX(m+2);r.fromBufferAttribute(t,y),s.fromBufferAttribute(t,b),a.fromBufferAttribute(t,g),h.subVectors(a,s),p.subVectors(r,s),h.cross(p),o.fromBufferAttribute(n,y),u.fromBufferAttribute(n,b),l.fromBufferAttribute(n,g),o.add(h),u.add(h),l.add(h),n.setXYZ(y,o.x,o.y,o.z),n.setXYZ(b,u.x,u.y,u.z),n.setXYZ(g,l.x,l.y,l.z)}else for(let m=0,v=t.count;m<v;m+=3)r.fromBufferAttribute(t,m+0),s.fromBufferAttribute(t,m+1),a.fromBufferAttribute(t,m+2),h.subVectors(a,s),p.subVectors(r,s),h.cross(p),n.setXYZ(m+0,h.x,h.y,h.z),n.setXYZ(m+1,h.x,h.y,h.z),n.setXYZ(m+2,h.x,h.y,h.z);this.normalizeNormals(),n.needsUpdate=!0}}normalizeNormals(){const e=this.attributes.normal;for(let t=0,n=e.count;t<n;t++)kt.fromBufferAttribute(e,t),kt.normalize(),e.setXYZ(t,kt.x,kt.y,kt.z)}toNonIndexed(){function e(o,u){const l=o.array,h=o.itemSize,p=o.normalized,m=new l.constructor(u.length*h);let v=0,y=0;for(let b=0,g=u.length;b<g;b++){o.isInterleavedBufferAttribute?v=u[b]*o.data.stride+o.offset:v=u[b]*h;for(let d=0;d<h;d++)m[y++]=l[v++]}return new _n(m,h,p)}if(this.index===null)return et("BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;const t=new bn,n=this.index.array,r=this.attributes;for(const o in r){const u=r[o],l=e(u,n);t.setAttribute(o,l)}const s=this.morphAttributes;for(const o in s){const u=[],l=s[o];for(let h=0,p=l.length;h<p;h++){const m=l[h],v=e(m,n);u.push(v)}t.morphAttributes[o]=u}t.morphTargetsRelative=this.morphTargetsRelative;const a=this.groups;for(let o=0,u=a.length;o<u;o++){const l=a[o];t.addGroup(l.start,l.count,l.materialIndex)}return t}toJSON(){const e={metadata:{version:4.7,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(e.uuid=this.uuid,e.type=this.type,this.name!==""&&(e.name=this.name),Object.keys(this.userData).length>0&&(e.userData=this.userData),this.parameters!==void 0){const u=this.parameters;for(const l in u)u[l]!==void 0&&(e[l]=u[l]);return e}e.data={attributes:{}};const t=this.index;t!==null&&(e.data.index={type:t.array.constructor.name,array:Array.prototype.slice.call(t.array)});const n=this.attributes;for(const u in n){const l=n[u];e.data.attributes[u]=l.toJSON(e.data)}const r={};let s=!1;for(const u in this.morphAttributes){const l=this.morphAttributes[u],h=[];for(let p=0,m=l.length;p<m;p++){const v=l[p];h.push(v.toJSON(e.data))}h.length>0&&(r[u]=h,s=!0)}s&&(e.data.morphAttributes=r,e.data.morphTargetsRelative=this.morphTargetsRelative);const a=this.groups;a.length>0&&(e.data.groups=JSON.parse(JSON.stringify(a)));const o=this.boundingSphere;return o!==null&&(e.data.boundingSphere=o.toJSON()),e}clone(){return new this.constructor().copy(this)}copy(e){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;const t={};this.name=e.name;const n=e.index;n!==null&&this.setIndex(n.clone());const r=e.attributes;for(const l in r){const h=r[l];this.setAttribute(l,h.clone(t))}const s=e.morphAttributes;for(const l in s){const h=[],p=s[l];for(let m=0,v=p.length;m<v;m++)h.push(p[m].clone(t));this.morphAttributes[l]=h}this.morphTargetsRelative=e.morphTargetsRelative;const a=e.groups;for(let l=0,h=a.length;l<h;l++){const p=a[l];this.addGroup(p.start,p.count,p.materialIndex)}const o=e.boundingBox;o!==null&&(this.boundingBox=o.clone());const u=e.boundingSphere;return u!==null&&(this.boundingSphere=u.clone()),this.drawRange.start=e.drawRange.start,this.drawRange.count=e.drawRange.count,this.userData=e.userData,this}dispose(){this.dispatchEvent({type:"dispose"})}}const Ho=new Ot,hi=new $u,Vr=new so,ko=new $,zr=new $,Gr=new $,Hr=new $,Hs=new $,kr=new $,Wo=new $,Wr=new $;let In=class extends ln{constructor(e=new bn,t=new Gl){super(),this.isMesh=!0,this.type="Mesh",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.count=1,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),e.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=e.morphTargetInfluences.slice()),e.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},e.morphTargetDictionary)),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}updateMorphTargets(){const t=this.geometry.morphAttributes,n=Object.keys(t);if(n.length>0){const r=t[n[0]];if(r!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let s=0,a=r.length;s<a;s++){const o=r[s].name||String(s);this.morphTargetInfluences.push(0),this.morphTargetDictionary[o]=s}}}}getVertexPosition(e,t){const n=this.geometry,r=n.attributes.position,s=n.morphAttributes.position,a=n.morphTargetsRelative;t.fromBufferAttribute(r,e);const o=this.morphTargetInfluences;if(s&&o){kr.set(0,0,0);for(let u=0,l=s.length;u<l;u++){const h=o[u],p=s[u];h!==0&&(Hs.fromBufferAttribute(p,e),a?kr.addScaledVector(Hs,h):kr.addScaledVector(Hs.sub(t),h))}t.add(kr)}return t}raycast(e,t){const n=this.geometry,r=this.material,s=this.matrixWorld;r!==void 0&&(n.boundingSphere===null&&n.computeBoundingSphere(),Vr.copy(n.boundingSphere),Vr.applyMatrix4(s),hi.copy(e.ray).recast(e.near),!(Vr.containsPoint(hi.origin)===!1&&(hi.intersectSphere(Vr,ko)===null||hi.origin.distanceToSquared(ko)>(e.far-e.near)**2))&&(Ho.copy(s).invert(),hi.copy(e.ray).applyMatrix4(Ho),!(n.boundingBox!==null&&hi.intersectsBox(n.boundingBox)===!1)&&this._computeIntersections(e,t,hi)))}_computeIntersections(e,t,n){let r;const s=this.geometry,a=this.material,o=s.index,u=s.attributes.position,l=s.attributes.uv,h=s.attributes.uv1,p=s.attributes.normal,m=s.groups,v=s.drawRange;if(o!==null)if(Array.isArray(a))for(let y=0,b=m.length;y<b;y++){const g=m[y],d=a[g.materialIndex],R=Math.max(g.start,v.start),C=Math.min(o.count,Math.min(g.start+g.count,v.start+v.count));for(let A=R,P=C;A<P;A+=3){const D=o.getX(A),I=o.getX(A+1),G=o.getX(A+2);r=Xr(this,d,e,n,l,h,p,D,I,G),r&&(r.faceIndex=Math.floor(A/3),r.face.materialIndex=g.materialIndex,t.push(r))}}else{const y=Math.max(0,v.start),b=Math.min(o.count,v.start+v.count);for(let g=y,d=b;g<d;g+=3){const R=o.getX(g),C=o.getX(g+1),A=o.getX(g+2);r=Xr(this,a,e,n,l,h,p,R,C,A),r&&(r.faceIndex=Math.floor(g/3),t.push(r))}}else if(u!==void 0)if(Array.isArray(a))for(let y=0,b=m.length;y<b;y++){const g=m[y],d=a[g.materialIndex],R=Math.max(g.start,v.start),C=Math.min(u.count,Math.min(g.start+g.count,v.start+v.count));for(let A=R,P=C;A<P;A+=3){const D=A,I=A+1,G=A+2;r=Xr(this,d,e,n,l,h,p,D,I,G),r&&(r.faceIndex=Math.floor(A/3),r.face.materialIndex=g.materialIndex,t.push(r))}}else{const y=Math.max(0,v.start),b=Math.min(u.count,v.start+v.count);for(let g=y,d=b;g<d;g+=3){const R=g,C=g+1,A=g+2;r=Xr(this,a,e,n,l,h,p,R,C,A),r&&(r.faceIndex=Math.floor(g/3),t.push(r))}}}};function nf(i,e,t,n,r,s,a,o){let u;if(e.side===on?u=n.intersectTriangle(a,s,r,!0,o):u=n.intersectTriangle(r,s,a,e.side===ai,o),u===null)return null;Wr.copy(o),Wr.applyMatrix4(i.matrixWorld);const l=t.ray.origin.distanceTo(Wr);return l<t.near||l>t.far?null:{distance:l,point:Wr.clone(),object:i}}function Xr(i,e,t,n,r,s,a,o,u,l){i.getVertexPosition(o,zr),i.getVertexPosition(u,Gr),i.getVertexPosition(l,Hr);const h=nf(i,e,t,n,zr,Gr,Hr,Wo);if(h){const p=new $;yn.getBarycoord(Wo,zr,Gr,Hr,p),r&&(h.uv=yn.getInterpolatedAttribute(r,o,u,l,p,new pt)),s&&(h.uv1=yn.getInterpolatedAttribute(s,o,u,l,p,new pt)),a&&(h.normal=yn.getInterpolatedAttribute(a,o,u,l,p,new $),h.normal.dot(n.direction)>0&&h.normal.multiplyScalar(-1));const m={a:o,b:u,c:l,normal:new $,materialIndex:0};yn.getNormal(zr,Gr,Hr,m.normal),h.face=m,h.barycoord=p}return h}class Ki extends bn{constructor(e=1,t=1,n=1,r=1,s=1,a=1){super(),this.type="BoxGeometry",this.parameters={width:e,height:t,depth:n,widthSegments:r,heightSegments:s,depthSegments:a};const o=this;r=Math.floor(r),s=Math.floor(s),a=Math.floor(a);const u=[],l=[],h=[],p=[];let m=0,v=0;y("z","y","x",-1,-1,n,t,e,a,s,0),y("z","y","x",1,-1,n,t,-e,a,s,1),y("x","z","y",1,1,e,n,t,r,a,2),y("x","z","y",1,-1,e,n,-t,r,a,3),y("x","y","z",1,-1,e,t,n,r,s,4),y("x","y","z",-1,-1,e,t,-n,r,s,5),this.setIndex(u),this.setAttribute("position",new gn(l,3)),this.setAttribute("normal",new gn(h,3)),this.setAttribute("uv",new gn(p,2));function y(b,g,d,R,C,A,P,D,I,G,x){const E=A/I,B=P/G,j=A/2,K=P/2,ie=D/2,oe=I+1,Q=G+1;let te=0,ue=0;const Ae=new $;for(let Ee=0;Ee<Q;Ee++){const Le=Ee*B-K;for(let at=0;at<oe;at++){const it=at*E-j;Ae[b]=it*R,Ae[g]=Le*C,Ae[d]=ie,l.push(Ae.x,Ae.y,Ae.z),Ae[b]=0,Ae[g]=0,Ae[d]=D>0?1:-1,h.push(Ae.x,Ae.y,Ae.z),p.push(at/I),p.push(1-Ee/G),te+=1}}for(let Ee=0;Ee<G;Ee++)for(let Le=0;Le<I;Le++){const at=m+Le+oe*Ee,it=m+Le+oe*(Ee+1),Ut=m+(Le+1)+oe*(Ee+1),Dt=m+(Le+1)+oe*Ee;u.push(at,it,Dt),u.push(it,Ut,Dt),ue+=6}o.addGroup(v,ue,x),v+=ue,m+=te}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new Ki(e.width,e.height,e.depth,e.widthSegments,e.heightSegments,e.depthSegments)}}function $i(i){const e={};for(const t in i){e[t]={};for(const n in i[t]){const r=i[t][n];r&&(r.isColor||r.isMatrix3||r.isMatrix4||r.isVector2||r.isVector3||r.isVector4||r.isTexture||r.isQuaternion)?r.isRenderTargetTexture?(et("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),e[t][n]=null):e[t][n]=r.clone():Array.isArray(r)?e[t][n]=r.slice():e[t][n]=r}}return e}function nn(i){const e={};for(let t=0;t<i.length;t++){const n=$i(i[t]);for(const r in n)e[r]=n[r]}return e}function rf(i){const e=[];for(let t=0;t<i.length;t++)e.push(i[t].clone());return e}function Wl(i){const e=i.getRenderTarget();return e===null?i.outputColorSpace:e.isXRRenderTarget===!0?e.texture.colorSpace:_t.workingColorSpace}const sf={clone:$i,merge:nn};var af=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,of=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`;class Fn extends Yi{constructor(e){super(),this.isShaderMaterial=!0,this.type="ShaderMaterial",this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=af,this.fragmentShader=of,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,e!==void 0&&this.setValues(e)}copy(e){return super.copy(e),this.fragmentShader=e.fragmentShader,this.vertexShader=e.vertexShader,this.uniforms=$i(e.uniforms),this.uniformsGroups=rf(e.uniformsGroups),this.defines=Object.assign({},e.defines),this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.fog=e.fog,this.lights=e.lights,this.clipping=e.clipping,this.extensions=Object.assign({},e.extensions),this.glslVersion=e.glslVersion,this.defaultAttributeValues=Object.assign({},e.defaultAttributeValues),this.index0AttributeName=e.index0AttributeName,this.uniformsNeedUpdate=e.uniformsNeedUpdate,this}toJSON(e){const t=super.toJSON(e);t.glslVersion=this.glslVersion,t.uniforms={};for(const r in this.uniforms){const a=this.uniforms[r].value;a&&a.isTexture?t.uniforms[r]={type:"t",value:a.toJSON(e).uuid}:a&&a.isColor?t.uniforms[r]={type:"c",value:a.getHex()}:a&&a.isVector2?t.uniforms[r]={type:"v2",value:a.toArray()}:a&&a.isVector3?t.uniforms[r]={type:"v3",value:a.toArray()}:a&&a.isVector4?t.uniforms[r]={type:"v4",value:a.toArray()}:a&&a.isMatrix3?t.uniforms[r]={type:"m3",value:a.toArray()}:a&&a.isMatrix4?t.uniforms[r]={type:"m4",value:a.toArray()}:t.uniforms[r]={value:a}}Object.keys(this.defines).length>0&&(t.defines=this.defines),t.vertexShader=this.vertexShader,t.fragmentShader=this.fragmentShader,t.lights=this.lights,t.clipping=this.clipping;const n={};for(const r in this.extensions)this.extensions[r]===!0&&(n[r]=!0);return Object.keys(n).length>0&&(t.extensions=n),t}}class Xl extends ln{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new Ot,this.projectionMatrix=new Ot,this.projectionMatrixInverse=new Ot,this.coordinateSystem=Cn,this._reversedDepth=!1}get reversedDepth(){return this._reversedDepth}copy(e,t){return super.copy(e,t),this.matrixWorldInverse.copy(e.matrixWorldInverse),this.projectionMatrix.copy(e.projectionMatrix),this.projectionMatrixInverse.copy(e.projectionMatrixInverse),this.coordinateSystem=e.coordinateSystem,this}getWorldDirection(e){return super.getWorldDirection(e).negate()}updateMatrixWorld(e){super.updateMatrixWorld(e),this.matrixWorldInverse.copy(this.matrixWorld).invert()}updateWorldMatrix(e,t){super.updateWorldMatrix(e,t),this.matrixWorldInverse.copy(this.matrixWorld).invert()}clone(){return new this.constructor().copy(this)}}const ii=new $,Xo=new pt,$o=new pt;class hn extends Xl{constructor(e=50,t=1,n=.1,r=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=e,this.zoom=1,this.near=n,this.far=r,this.focus=10,this.aspect=t,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.fov=e.fov,this.zoom=e.zoom,this.near=e.near,this.far=e.far,this.focus=e.focus,this.aspect=e.aspect,this.view=e.view===null?null:Object.assign({},e.view),this.filmGauge=e.filmGauge,this.filmOffset=e.filmOffset,this}setFocalLength(e){const t=.5*this.getFilmHeight()/e;this.fov=ka*2*Math.atan(t),this.updateProjectionMatrix()}getFocalLength(){const e=Math.tan(Ss*.5*this.fov);return .5*this.getFilmHeight()/e}getEffectiveFOV(){return ka*2*Math.atan(Math.tan(Ss*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(e,t,n){ii.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),t.set(ii.x,ii.y).multiplyScalar(-e/ii.z),ii.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),n.set(ii.x,ii.y).multiplyScalar(-e/ii.z)}getViewSize(e,t){return this.getViewBounds(e,Xo,$o),t.subVectors($o,Xo)}setViewOffset(e,t,n,r,s,a){this.aspect=e/t,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=n,this.view.offsetY=r,this.view.width=s,this.view.height=a,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=this.near;let t=e*Math.tan(Ss*.5*this.fov)/this.zoom,n=2*t,r=this.aspect*n,s=-.5*r;const a=this.view;if(this.view!==null&&this.view.enabled){const u=a.fullWidth,l=a.fullHeight;s+=a.offsetX*r/u,t-=a.offsetY*n/l,r*=a.width/u,n*=a.height/l}const o=this.filmOffset;o!==0&&(s+=e*o/this.getFilmWidth()),this.projectionMatrix.makePerspective(s,s+r,t,t-n,e,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const t=super.toJSON(e);return t.object.fov=this.fov,t.object.zoom=this.zoom,t.object.near=this.near,t.object.far=this.far,t.object.focus=this.focus,t.object.aspect=this.aspect,this.view!==null&&(t.object.view=Object.assign({},this.view)),t.object.filmGauge=this.filmGauge,t.object.filmOffset=this.filmOffset,t}}const Oi=-90,Bi=1;class lf extends ln{constructor(e,t,n){super(),this.type="CubeCamera",this.renderTarget=n,this.coordinateSystem=null,this.activeMipmapLevel=0;const r=new hn(Oi,Bi,e,t);r.layers=this.layers,this.add(r);const s=new hn(Oi,Bi,e,t);s.layers=this.layers,this.add(s);const a=new hn(Oi,Bi,e,t);a.layers=this.layers,this.add(a);const o=new hn(Oi,Bi,e,t);o.layers=this.layers,this.add(o);const u=new hn(Oi,Bi,e,t);u.layers=this.layers,this.add(u);const l=new hn(Oi,Bi,e,t);l.layers=this.layers,this.add(l)}updateCoordinateSystem(){const e=this.coordinateSystem,t=this.children.concat(),[n,r,s,a,o,u]=t;for(const l of t)this.remove(l);if(e===Cn)n.up.set(0,1,0),n.lookAt(1,0,0),r.up.set(0,1,0),r.lookAt(-1,0,0),s.up.set(0,0,-1),s.lookAt(0,1,0),a.up.set(0,0,1),a.lookAt(0,-1,0),o.up.set(0,1,0),o.lookAt(0,0,1),u.up.set(0,1,0),u.lookAt(0,0,-1);else if(e===is)n.up.set(0,-1,0),n.lookAt(-1,0,0),r.up.set(0,-1,0),r.lookAt(1,0,0),s.up.set(0,0,1),s.lookAt(0,1,0),a.up.set(0,0,-1),a.lookAt(0,-1,0),o.up.set(0,-1,0),o.lookAt(0,0,1),u.up.set(0,-1,0),u.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+e);for(const l of t)this.add(l),l.updateMatrixWorld()}update(e,t){this.parent===null&&this.updateMatrixWorld();const{renderTarget:n,activeMipmapLevel:r}=this;this.coordinateSystem!==e.coordinateSystem&&(this.coordinateSystem=e.coordinateSystem,this.updateCoordinateSystem());const[s,a,o,u,l,h]=this.children,p=e.getRenderTarget(),m=e.getActiveCubeFace(),v=e.getActiveMipmapLevel(),y=e.xr.enabled;e.xr.enabled=!1;const b=n.texture.generateMipmaps;n.texture.generateMipmaps=!1,e.setRenderTarget(n,0,r),e.render(t,s),e.setRenderTarget(n,1,r),e.render(t,a),e.setRenderTarget(n,2,r),e.render(t,o),e.setRenderTarget(n,3,r),e.render(t,u),e.setRenderTarget(n,4,r),e.render(t,l),n.texture.generateMipmaps=b,e.setRenderTarget(n,5,r),e.render(t,h),e.setRenderTarget(p,m,v),e.xr.enabled=y,n.texture.needsPMREMUpdate=!0}}class $l extends rn{constructor(e=[],t=Si,n,r,s,a,o,u,l,h){super(e,t,n,r,s,a,o,u,l,h),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(e){this.image=e}}class ql extends Dn{constructor(e=1,t={}){super(e,e,t),this.isWebGLCubeRenderTarget=!0;const n={width:e,height:e,depth:1},r=[n,n,n,n,n,n];this.texture=new $l(r),this._setTextureOptions(t),this.texture.isRenderTargetTexture=!0}fromEquirectangularTexture(e,t){this.texture.type=t.type,this.texture.colorSpace=t.colorSpace,this.texture.generateMipmaps=t.generateMipmaps,this.texture.minFilter=t.minFilter,this.texture.magFilter=t.magFilter;const n={uniforms:{tEquirect:{value:null}},vertexShader:`

				varying vec3 vWorldDirection;

				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

				}

				void main() {

					vWorldDirection = transformDirection( position, modelMatrix );

					#include <begin_vertex>
					#include <project_vertex>

				}
			`,fragmentShader:`

				uniform sampler2D tEquirect;

				varying vec3 vWorldDirection;

				#include <common>

				void main() {

					vec3 direction = normalize( vWorldDirection );

					vec2 sampleUV = equirectUv( direction );

					gl_FragColor = texture2D( tEquirect, sampleUV );

				}
			`},r=new Ki(5,5,5),s=new Fn({name:"CubemapFromEquirect",uniforms:$i(n.uniforms),vertexShader:n.vertexShader,fragmentShader:n.fragmentShader,side:on,blending:$n});s.uniforms.tEquirect.value=t;const a=new In(r,s),o=t.minFilter;return t.minFilter===xi&&(t.minFilter=Qt),new lf(1,10,this).update(e,a),t.minFilter=o,a.geometry.dispose(),a.material.dispose(),this}clear(e,t=!0,n=!0,r=!0){const s=e.getRenderTarget();for(let a=0;a<6;a++)e.setRenderTarget(this,a),e.clear(t,n,r);e.setRenderTarget(s)}}class $r extends ln{constructor(){super(),this.isGroup=!0,this.type="Group"}}const cf={type:"move"};class ks{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new $r,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new $r,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new $,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new $),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new $r,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new $,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new $),this._grip}dispatchEvent(e){return this._targetRay!==null&&this._targetRay.dispatchEvent(e),this._grip!==null&&this._grip.dispatchEvent(e),this._hand!==null&&this._hand.dispatchEvent(e),this}connect(e){if(e&&e.hand){const t=this._hand;if(t)for(const n of e.hand.values())this._getHandJoint(t,n)}return this.dispatchEvent({type:"connected",data:e}),this}disconnect(e){return this.dispatchEvent({type:"disconnected",data:e}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(e,t,n){let r=null,s=null,a=null;const o=this._targetRay,u=this._grip,l=this._hand;if(e&&t.session.visibilityState!=="visible-blurred"){if(l&&e.hand){a=!0;for(const b of e.hand.values()){const g=t.getJointPose(b,n),d=this._getHandJoint(l,b);g!==null&&(d.matrix.fromArray(g.transform.matrix),d.matrix.decompose(d.position,d.rotation,d.scale),d.matrixWorldNeedsUpdate=!0,d.jointRadius=g.radius),d.visible=g!==null}const h=l.joints["index-finger-tip"],p=l.joints["thumb-tip"],m=h.position.distanceTo(p.position),v=.02,y=.005;l.inputState.pinching&&m>v+y?(l.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:e.handedness,target:this})):!l.inputState.pinching&&m<=v-y&&(l.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:e.handedness,target:this}))}else u!==null&&e.gripSpace&&(s=t.getPose(e.gripSpace,n),s!==null&&(u.matrix.fromArray(s.transform.matrix),u.matrix.decompose(u.position,u.rotation,u.scale),u.matrixWorldNeedsUpdate=!0,s.linearVelocity?(u.hasLinearVelocity=!0,u.linearVelocity.copy(s.linearVelocity)):u.hasLinearVelocity=!1,s.angularVelocity?(u.hasAngularVelocity=!0,u.angularVelocity.copy(s.angularVelocity)):u.hasAngularVelocity=!1));o!==null&&(r=t.getPose(e.targetRaySpace,n),r===null&&s!==null&&(r=s),r!==null&&(o.matrix.fromArray(r.transform.matrix),o.matrix.decompose(o.position,o.rotation,o.scale),o.matrixWorldNeedsUpdate=!0,r.linearVelocity?(o.hasLinearVelocity=!0,o.linearVelocity.copy(r.linearVelocity)):o.hasLinearVelocity=!1,r.angularVelocity?(o.hasAngularVelocity=!0,o.angularVelocity.copy(r.angularVelocity)):o.hasAngularVelocity=!1,this.dispatchEvent(cf)))}return o!==null&&(o.visible=r!==null),u!==null&&(u.visible=s!==null),l!==null&&(l.visible=a!==null),this}_getHandJoint(e,t){if(e.joints[t.jointName]===void 0){const n=new $r;n.matrixAutoUpdate=!1,n.visible=!1,e.joints[t.jointName]=n,e.add(n)}return e.joints[t.jointName]}}class uf extends ln{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.backgroundRotation=new Un,this.environmentIntensity=1,this.environmentRotation=new Un,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(e,t){return super.copy(e,t),e.background!==null&&(this.background=e.background.clone()),e.environment!==null&&(this.environment=e.environment.clone()),e.fog!==null&&(this.fog=e.fog.clone()),this.backgroundBlurriness=e.backgroundBlurriness,this.backgroundIntensity=e.backgroundIntensity,this.backgroundRotation.copy(e.backgroundRotation),this.environmentIntensity=e.environmentIntensity,this.environmentRotation.copy(e.environmentRotation),e.overrideMaterial!==null&&(this.overrideMaterial=e.overrideMaterial.clone()),this.matrixAutoUpdate=e.matrixAutoUpdate,this}toJSON(e){const t=super.toJSON(e);return this.fog!==null&&(t.object.fog=this.fog.toJSON()),this.backgroundBlurriness>0&&(t.object.backgroundBlurriness=this.backgroundBlurriness),this.backgroundIntensity!==1&&(t.object.backgroundIntensity=this.backgroundIntensity),t.object.backgroundRotation=this.backgroundRotation.toArray(),this.environmentIntensity!==1&&(t.object.environmentIntensity=this.environmentIntensity),t.object.environmentRotation=this.environmentRotation.toArray(),t}}class ff extends rn{constructor(e=null,t=1,n=1,r,s,a,o,u,l=Xt,h=Xt,p,m){super(null,a,o,u,l,h,r,s,p,m),this.isDataTexture=!0,this.image={data:e,width:t,height:n},this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}const Ws=new $,hf=new $,df=new st;class _i{constructor(e=new $(1,0,0),t=0){this.isPlane=!0,this.normal=e,this.constant=t}set(e,t){return this.normal.copy(e),this.constant=t,this}setComponents(e,t,n,r){return this.normal.set(e,t,n),this.constant=r,this}setFromNormalAndCoplanarPoint(e,t){return this.normal.copy(e),this.constant=-t.dot(this.normal),this}setFromCoplanarPoints(e,t,n){const r=Ws.subVectors(n,t).cross(hf.subVectors(e,t)).normalize();return this.setFromNormalAndCoplanarPoint(r,e),this}copy(e){return this.normal.copy(e.normal),this.constant=e.constant,this}normalize(){const e=1/this.normal.length();return this.normal.multiplyScalar(e),this.constant*=e,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(e){return this.normal.dot(e)+this.constant}distanceToSphere(e){return this.distanceToPoint(e.center)-e.radius}projectPoint(e,t){return t.copy(e).addScaledVector(this.normal,-this.distanceToPoint(e))}intersectLine(e,t){const n=e.delta(Ws),r=this.normal.dot(n);if(r===0)return this.distanceToPoint(e.start)===0?t.copy(e.start):null;const s=-(e.start.dot(this.normal)+this.constant)/r;return s<0||s>1?null:t.copy(e.start).addScaledVector(n,s)}intersectsLine(e){const t=this.distanceToPoint(e.start),n=this.distanceToPoint(e.end);return t<0&&n>0||n<0&&t>0}intersectsBox(e){return e.intersectsPlane(this)}intersectsSphere(e){return e.intersectsPlane(this)}coplanarPoint(e){return e.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(e,t){const n=t||df.getNormalMatrix(e),r=this.coplanarPoint(Ws).applyMatrix4(e),s=this.normal.applyMatrix3(n).normalize();return this.constant=-r.dot(s),this}translate(e){return this.constant-=e.dot(this.normal),this}equals(e){return e.normal.equals(this.normal)&&e.constant===this.constant}clone(){return new this.constructor().copy(this)}}const di=new so,pf=new pt(.5,.5),qr=new $;class ao{constructor(e=new _i,t=new _i,n=new _i,r=new _i,s=new _i,a=new _i){this.planes=[e,t,n,r,s,a]}set(e,t,n,r,s,a){const o=this.planes;return o[0].copy(e),o[1].copy(t),o[2].copy(n),o[3].copy(r),o[4].copy(s),o[5].copy(a),this}copy(e){const t=this.planes;for(let n=0;n<6;n++)t[n].copy(e.planes[n]);return this}setFromProjectionMatrix(e,t=Cn,n=!1){const r=this.planes,s=e.elements,a=s[0],o=s[1],u=s[2],l=s[3],h=s[4],p=s[5],m=s[6],v=s[7],y=s[8],b=s[9],g=s[10],d=s[11],R=s[12],C=s[13],A=s[14],P=s[15];if(r[0].setComponents(l-a,v-h,d-y,P-R).normalize(),r[1].setComponents(l+a,v+h,d+y,P+R).normalize(),r[2].setComponents(l+o,v+p,d+b,P+C).normalize(),r[3].setComponents(l-o,v-p,d-b,P-C).normalize(),n)r[4].setComponents(u,m,g,A).normalize(),r[5].setComponents(l-u,v-m,d-g,P-A).normalize();else if(r[4].setComponents(l-u,v-m,d-g,P-A).normalize(),t===Cn)r[5].setComponents(l+u,v+m,d+g,P+A).normalize();else if(t===is)r[5].setComponents(u,m,g,A).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+t);return this}intersectsObject(e){if(e.boundingSphere!==void 0)e.boundingSphere===null&&e.computeBoundingSphere(),di.copy(e.boundingSphere).applyMatrix4(e.matrixWorld);else{const t=e.geometry;t.boundingSphere===null&&t.computeBoundingSphere(),di.copy(t.boundingSphere).applyMatrix4(e.matrixWorld)}return this.intersectsSphere(di)}intersectsSprite(e){di.center.set(0,0,0);const t=pf.distanceTo(e.center);return di.radius=.7071067811865476+t,di.applyMatrix4(e.matrixWorld),this.intersectsSphere(di)}intersectsSphere(e){const t=this.planes,n=e.center,r=-e.radius;for(let s=0;s<6;s++)if(t[s].distanceToPoint(n)<r)return!1;return!0}intersectsBox(e){const t=this.planes;for(let n=0;n<6;n++){const r=t[n];if(qr.x=r.normal.x>0?e.max.x:e.min.x,qr.y=r.normal.y>0?e.max.y:e.min.y,qr.z=r.normal.z>0?e.max.z:e.min.z,r.distanceToPoint(qr)<0)return!1}return!0}containsPoint(e){const t=this.planes;for(let n=0;n<6;n++)if(t[n].distanceToPoint(e)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}}class mr extends rn{constructor(e,t,n=Ln,r,s,a,o=Xt,u=Xt,l,h=Kn,p=1){if(h!==Kn&&h!==Mi)throw new Error("DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat");const m={width:e,height:t,depth:p};super(m,r,s,a,o,u,h,n,l),this.isDepthTexture=!0,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(e){return super.copy(e),this.source=new ro(Object.assign({},e.image)),this.compareFunction=e.compareFunction,this}toJSON(e){const t=super.toJSON(e);return this.compareFunction!==null&&(t.compareFunction=this.compareFunction),t}}class mf extends mr{constructor(e,t=Ln,n=Si,r,s,a=Xt,o=Xt,u,l=Kn){const h={width:e,height:e,depth:1},p=[h,h,h,h,h,h];super(e,e,t,n,r,s,a,o,u,l),this.image=p,this.isCubeDepthTexture=!0,this.isCubeTexture=!0}get images(){return this.image}set images(e){this.image=e}}class Yl extends rn{constructor(e=null){super(),this.sourceTexture=e,this.isExternalTexture=!0}copy(e){return super.copy(e),this.sourceTexture=e.sourceTexture,this}}class oo extends bn{constructor(e=[],t=[],n=1,r=0){super(),this.type="PolyhedronGeometry",this.parameters={vertices:e,indices:t,radius:n,detail:r};const s=[],a=[];o(r),l(n),h(),this.setAttribute("position",new gn(s,3)),this.setAttribute("normal",new gn(s.slice(),3)),this.setAttribute("uv",new gn(a,2)),r===0?this.computeVertexNormals():this.normalizeNormals();function o(R){const C=new $,A=new $,P=new $;for(let D=0;D<t.length;D+=3)v(t[D+0],C),v(t[D+1],A),v(t[D+2],P),u(C,A,P,R)}function u(R,C,A,P){const D=P+1,I=[];for(let G=0;G<=D;G++){I[G]=[];const x=R.clone().lerp(A,G/D),E=C.clone().lerp(A,G/D),B=D-G;for(let j=0;j<=B;j++)j===0&&G===D?I[G][j]=x:I[G][j]=x.clone().lerp(E,j/B)}for(let G=0;G<D;G++)for(let x=0;x<2*(D-G)-1;x++){const E=Math.floor(x/2);x%2===0?(m(I[G][E+1]),m(I[G+1][E]),m(I[G][E])):(m(I[G][E+1]),m(I[G+1][E+1]),m(I[G+1][E]))}}function l(R){const C=new $;for(let A=0;A<s.length;A+=3)C.x=s[A+0],C.y=s[A+1],C.z=s[A+2],C.normalize().multiplyScalar(R),s[A+0]=C.x,s[A+1]=C.y,s[A+2]=C.z}function h(){const R=new $;for(let C=0;C<s.length;C+=3){R.x=s[C+0],R.y=s[C+1],R.z=s[C+2];const A=g(R)/2/Math.PI+.5,P=d(R)/Math.PI+.5;a.push(A,1-P)}y(),p()}function p(){for(let R=0;R<a.length;R+=6){const C=a[R+0],A=a[R+2],P=a[R+4],D=Math.max(C,A,P),I=Math.min(C,A,P);D>.9&&I<.1&&(C<.2&&(a[R+0]+=1),A<.2&&(a[R+2]+=1),P<.2&&(a[R+4]+=1))}}function m(R){s.push(R.x,R.y,R.z)}function v(R,C){const A=R*3;C.x=e[A+0],C.y=e[A+1],C.z=e[A+2]}function y(){const R=new $,C=new $,A=new $,P=new $,D=new pt,I=new pt,G=new pt;for(let x=0,E=0;x<s.length;x+=9,E+=6){R.set(s[x+0],s[x+1],s[x+2]),C.set(s[x+3],s[x+4],s[x+5]),A.set(s[x+6],s[x+7],s[x+8]),D.set(a[E+0],a[E+1]),I.set(a[E+2],a[E+3]),G.set(a[E+4],a[E+5]),P.copy(R).add(C).add(A).divideScalar(3);const B=g(P);b(D,E+0,R,B),b(I,E+2,C,B),b(G,E+4,A,B)}}function b(R,C,A,P){P<0&&R.x===1&&(a[C]=R.x-1),A.x===0&&A.z===0&&(a[C]=P/2/Math.PI+.5)}function g(R){return Math.atan2(R.z,-R.x)}function d(R){return Math.atan2(-R.y,Math.sqrt(R.x*R.x+R.z*R.z))}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new oo(e.vertices,e.indices,e.radius,e.detail)}}class lo extends oo{constructor(e=1,t=0){const n=(1+Math.sqrt(5))/2,r=[-1,n,0,1,n,0,-1,-n,0,1,-n,0,0,-1,n,0,1,n,0,-1,-n,0,1,-n,n,0,-1,n,0,1,-n,0,-1,-n,0,1],s=[0,11,5,0,5,1,0,1,7,0,7,10,0,10,11,1,5,9,5,11,4,11,10,2,10,7,6,7,1,8,3,9,4,3,4,2,3,2,6,3,6,8,3,8,9,4,9,5,2,4,11,6,2,10,8,6,7,9,8,1];super(r,s,e,t),this.type="IcosahedronGeometry",this.parameters={radius:e,detail:t}}static fromJSON(e){return new lo(e.radius,e.detail)}}class as extends bn{constructor(e=1,t=1,n=1,r=1){super(),this.type="PlaneGeometry",this.parameters={width:e,height:t,widthSegments:n,heightSegments:r};const s=e/2,a=t/2,o=Math.floor(n),u=Math.floor(r),l=o+1,h=u+1,p=e/o,m=t/u,v=[],y=[],b=[],g=[];for(let d=0;d<h;d++){const R=d*m-a;for(let C=0;C<l;C++){const A=C*p-s;y.push(A,-R,0),b.push(0,0,1),g.push(C/o),g.push(1-d/u)}}for(let d=0;d<u;d++)for(let R=0;R<o;R++){const C=R+l*d,A=R+l*(d+1),P=R+1+l*(d+1),D=R+1+l*d;v.push(C,A,D),v.push(A,P,D)}this.setIndex(v),this.setAttribute("position",new gn(y,3)),this.setAttribute("normal",new gn(b,3)),this.setAttribute("uv",new gn(g,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new as(e.width,e.height,e.widthSegments,e.heightSegments)}}class _f extends Fn{constructor(e){super(e),this.isRawShaderMaterial=!0,this.type="RawShaderMaterial"}}class gf extends Yi{constructor(e){super(),this.isMeshNormalMaterial=!0,this.type="MeshNormalMaterial",this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=to,this.normalScale=new pt(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.flatShading=!1,this.setValues(e)}copy(e){return super.copy(e),this.bumpMap=e.bumpMap,this.bumpScale=e.bumpScale,this.normalMap=e.normalMap,this.normalMapType=e.normalMapType,this.normalScale.copy(e.normalScale),this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.flatShading=e.flatShading,this}}class qo extends Yi{constructor(e){super(),this.isMeshLambertMaterial=!0,this.type="MeshLambertMaterial",this.color=new yt(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.emissive=new yt(0),this.emissiveIntensity=1,this.emissiveMap=null,this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=to,this.normalScale=new pt(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new Un,this.combine=Ya,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.flatShading=!1,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.emissive.copy(e.emissive),this.emissiveMap=e.emissiveMap,this.emissiveIntensity=e.emissiveIntensity,this.bumpMap=e.bumpMap,this.bumpScale=e.bumpScale,this.normalMap=e.normalMap,this.normalMapType=e.normalMapType,this.normalScale.copy(e.normalScale),this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.flatShading=e.flatShading,this.fog=e.fog,this}}class vf extends Yi{constructor(e){super(),this.isMeshDepthMaterial=!0,this.type="MeshDepthMaterial",this.depthPacking=Ru,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(e)}copy(e){return super.copy(e),this.depthPacking=e.depthPacking,this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this}}class xf extends Yi{constructor(e){super(),this.isMeshDistanceMaterial=!0,this.type="MeshDistanceMaterial",this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(e)}copy(e){return super.copy(e),this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this}}class Mf extends ln{constructor(e,t=1){super(),this.isLight=!0,this.type="Light",this.color=new yt(e),this.intensity=t}dispose(){this.dispatchEvent({type:"dispose"})}copy(e,t){return super.copy(e,t),this.color.copy(e.color),this.intensity=e.intensity,this}toJSON(e){const t=super.toJSON(e);return t.object.color=this.color.getHex(),t.object.intensity=this.intensity,t}}const Xs=new Ot,Yo=new $,Ko=new $;class Sf{constructor(e){this.camera=e,this.intensity=1,this.bias=0,this.normalBias=0,this.radius=1,this.blurSamples=8,this.mapSize=new pt(512,512),this.mapType=dn,this.map=null,this.mapPass=null,this.matrix=new Ot,this.autoUpdate=!0,this.needsUpdate=!1,this._frustum=new ao,this._frameExtents=new pt(1,1),this._viewportCount=1,this._viewports=[new Nt(0,0,1,1)]}getViewportCount(){return this._viewportCount}getFrustum(){return this._frustum}updateMatrices(e){const t=this.camera,n=this.matrix;Yo.setFromMatrixPosition(e.matrixWorld),t.position.copy(Yo),Ko.setFromMatrixPosition(e.target.matrixWorld),t.lookAt(Ko),t.updateMatrixWorld(),Xs.multiplyMatrices(t.projectionMatrix,t.matrixWorldInverse),this._frustum.setFromProjectionMatrix(Xs,t.coordinateSystem,t.reversedDepth),t.reversedDepth?n.set(.5,0,0,.5,0,.5,0,.5,0,0,1,0,0,0,0,1):n.set(.5,0,0,.5,0,.5,0,.5,0,0,.5,.5,0,0,0,1),n.multiply(Xs)}getViewport(e){return this._viewports[e]}getFrameExtents(){return this._frameExtents}dispose(){this.map&&this.map.dispose(),this.mapPass&&this.mapPass.dispose()}copy(e){return this.camera=e.camera.clone(),this.intensity=e.intensity,this.bias=e.bias,this.radius=e.radius,this.autoUpdate=e.autoUpdate,this.needsUpdate=e.needsUpdate,this.normalBias=e.normalBias,this.blurSamples=e.blurSamples,this.mapSize.copy(e.mapSize),this}clone(){return new this.constructor().copy(this)}toJSON(){const e={};return this.intensity!==1&&(e.intensity=this.intensity),this.bias!==0&&(e.bias=this.bias),this.normalBias!==0&&(e.normalBias=this.normalBias),this.radius!==1&&(e.radius=this.radius),(this.mapSize.x!==512||this.mapSize.y!==512)&&(e.mapSize=this.mapSize.toArray()),e.camera=this.camera.toJSON(!1).object,delete e.camera.matrix,e}}class yf extends Sf{constructor(){super(new hn(90,1,.5,500)),this.isPointLightShadow=!0}}class Ef extends Mf{constructor(e,t,n=0,r=2){super(e,t),this.isPointLight=!0,this.type="PointLight",this.distance=n,this.decay=r,this.shadow=new yf}get power(){return this.intensity*4*Math.PI}set power(e){this.intensity=e/(4*Math.PI)}dispose(){super.dispose(),this.shadow.dispose()}copy(e,t){return super.copy(e,t),this.distance=e.distance,this.decay=e.decay,this.shadow=e.shadow.clone(),this}toJSON(e){const t=super.toJSON(e);return t.object.distance=this.distance,t.object.decay=this.decay,t.object.shadow=this.shadow.toJSON(),t}}class Kl extends Xl{constructor(e=-1,t=1,n=1,r=-1,s=.1,a=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=e,this.right=t,this.top=n,this.bottom=r,this.near=s,this.far=a,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.left=e.left,this.right=e.right,this.top=e.top,this.bottom=e.bottom,this.near=e.near,this.far=e.far,this.zoom=e.zoom,this.view=e.view===null?null:Object.assign({},e.view),this}setViewOffset(e,t,n,r,s,a){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=n,this.view.offsetY=r,this.view.width=s,this.view.height=a,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=(this.right-this.left)/(2*this.zoom),t=(this.top-this.bottom)/(2*this.zoom),n=(this.right+this.left)/2,r=(this.top+this.bottom)/2;let s=n-e,a=n+e,o=r+t,u=r-t;if(this.view!==null&&this.view.enabled){const l=(this.right-this.left)/this.view.fullWidth/this.zoom,h=(this.top-this.bottom)/this.view.fullHeight/this.zoom;s+=l*this.view.offsetX,a=s+l*this.view.width,o-=h*this.view.offsetY,u=o-h*this.view.height}this.projectionMatrix.makeOrthographic(s,a,o,u,this.near,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const t=super.toJSON(e);return t.object.zoom=this.zoom,t.object.left=this.left,t.object.right=this.right,t.object.top=this.top,t.object.bottom=this.bottom,t.object.near=this.near,t.object.far=this.far,this.view!==null&&(t.object.view=Object.assign({},this.view)),t}}class bf extends hn{constructor(e=[]){super(),this.isArrayCamera=!0,this.isMultiViewCamera=!1,this.cameras=e}}function jo(i,e,t,n){const r=Tf(n);switch(t){case Il:return i*e;case Nl:return i*e/r.components*r.byteLength;case Ja:return i*e/r.components*r.byteLength;case Wi:return i*e*2/r.components*r.byteLength;case Qa:return i*e*2/r.components*r.byteLength;case Fl:return i*e*3/r.components*r.byteLength;case En:return i*e*4/r.components*r.byteLength;case eo:return i*e*4/r.components*r.byteLength;case Zr:case Jr:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*8;case Qr:case es:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case ha:case pa:return Math.max(i,16)*Math.max(e,8)/4;case fa:case da:return Math.max(i,8)*Math.max(e,8)/2;case ma:case _a:case va:case xa:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*8;case ga:case Ma:case Sa:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case ya:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case Ea:return Math.floor((i+4)/5)*Math.floor((e+3)/4)*16;case ba:return Math.floor((i+4)/5)*Math.floor((e+4)/5)*16;case Ta:return Math.floor((i+5)/6)*Math.floor((e+4)/5)*16;case Aa:return Math.floor((i+5)/6)*Math.floor((e+5)/6)*16;case wa:return Math.floor((i+7)/8)*Math.floor((e+4)/5)*16;case Ra:return Math.floor((i+7)/8)*Math.floor((e+5)/6)*16;case Ca:return Math.floor((i+7)/8)*Math.floor((e+7)/8)*16;case Pa:return Math.floor((i+9)/10)*Math.floor((e+4)/5)*16;case Da:return Math.floor((i+9)/10)*Math.floor((e+5)/6)*16;case La:return Math.floor((i+9)/10)*Math.floor((e+7)/8)*16;case Ua:return Math.floor((i+9)/10)*Math.floor((e+9)/10)*16;case Ia:return Math.floor((i+11)/12)*Math.floor((e+9)/10)*16;case Fa:return Math.floor((i+11)/12)*Math.floor((e+11)/12)*16;case Na:case Oa:case Ba:return Math.ceil(i/4)*Math.ceil(e/4)*16;case Va:case za:return Math.ceil(i/4)*Math.ceil(e/4)*8;case Ga:case Ha:return Math.ceil(i/4)*Math.ceil(e/4)*16}throw new Error(`Unable to determine texture byte length for ${t} format.`)}function Tf(i){switch(i){case dn:case Pl:return{byteLength:1,components:1};case hr:case Dl:case Yn:return{byteLength:2,components:1};case ja:case Za:return{byteLength:2,components:4};case Ln:case Ka:case Rn:return{byteLength:4,components:1};case Ll:case Ul:return{byteLength:4,components:3}}throw new Error(`Unknown texture type ${i}.`)}typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:qa}}));typeof window<"u"&&(window.__THREE__?et("WARNING: Multiple instances of Three.js being imported."):window.__THREE__=qa);function jl(){let i=null,e=!1,t=null,n=null;function r(s,a){t(s,a),n=i.requestAnimationFrame(r)}return{start:function(){e!==!0&&t!==null&&(n=i.requestAnimationFrame(r),e=!0)},stop:function(){i.cancelAnimationFrame(n),e=!1},setAnimationLoop:function(s){t=s},setContext:function(s){i=s}}}function Af(i){const e=new WeakMap;function t(o,u){const l=o.array,h=o.usage,p=l.byteLength,m=i.createBuffer();i.bindBuffer(u,m),i.bufferData(u,l,h),o.onUploadCallback();let v;if(l instanceof Float32Array)v=i.FLOAT;else if(typeof Float16Array<"u"&&l instanceof Float16Array)v=i.HALF_FLOAT;else if(l instanceof Uint16Array)o.isFloat16BufferAttribute?v=i.HALF_FLOAT:v=i.UNSIGNED_SHORT;else if(l instanceof Int16Array)v=i.SHORT;else if(l instanceof Uint32Array)v=i.UNSIGNED_INT;else if(l instanceof Int32Array)v=i.INT;else if(l instanceof Int8Array)v=i.BYTE;else if(l instanceof Uint8Array)v=i.UNSIGNED_BYTE;else if(l instanceof Uint8ClampedArray)v=i.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+l);return{buffer:m,type:v,bytesPerElement:l.BYTES_PER_ELEMENT,version:o.version,size:p}}function n(o,u,l){const h=u.array,p=u.updateRanges;if(i.bindBuffer(l,o),p.length===0)i.bufferSubData(l,0,h);else{p.sort((v,y)=>v.start-y.start);let m=0;for(let v=1;v<p.length;v++){const y=p[m],b=p[v];b.start<=y.start+y.count+1?y.count=Math.max(y.count,b.start+b.count-y.start):(++m,p[m]=b)}p.length=m+1;for(let v=0,y=p.length;v<y;v++){const b=p[v];i.bufferSubData(l,b.start*h.BYTES_PER_ELEMENT,h,b.start,b.count)}u.clearUpdateRanges()}u.onUploadCallback()}function r(o){return o.isInterleavedBufferAttribute&&(o=o.data),e.get(o)}function s(o){o.isInterleavedBufferAttribute&&(o=o.data);const u=e.get(o);u&&(i.deleteBuffer(u.buffer),e.delete(o))}function a(o,u){if(o.isInterleavedBufferAttribute&&(o=o.data),o.isGLBufferAttribute){const h=e.get(o);(!h||h.version<o.version)&&e.set(o,{buffer:o.buffer,type:o.type,bytesPerElement:o.elementSize,version:o.version});return}const l=e.get(o);if(l===void 0)e.set(o,t(o,u));else if(l.version<o.version){if(l.size!==o.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");n(l.buffer,o,u),l.version=o.version}}return{get:r,remove:s,update:a}}var wf=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,Rf=`#ifdef USE_ALPHAHASH
	const float ALPHA_HASH_SCALE = 0.05;
	float hash2D( vec2 value ) {
		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );
	}
	float hash3D( vec3 value ) {
		return hash2D( vec2( hash2D( value.xy ), value.z ) );
	}
	float getAlphaHashThreshold( vec3 position ) {
		float maxDeriv = max(
			length( dFdx( position.xyz ) ),
			length( dFdy( position.xyz ) )
		);
		float pixScale = 1.0 / ( ALPHA_HASH_SCALE * maxDeriv );
		vec2 pixScales = vec2(
			exp2( floor( log2( pixScale ) ) ),
			exp2( ceil( log2( pixScale ) ) )
		);
		vec2 alpha = vec2(
			hash3D( floor( pixScales.x * position.xyz ) ),
			hash3D( floor( pixScales.y * position.xyz ) )
		);
		float lerpFactor = fract( log2( pixScale ) );
		float x = ( 1.0 - lerpFactor ) * alpha.x + lerpFactor * alpha.y;
		float a = min( lerpFactor, 1.0 - lerpFactor );
		vec3 cases = vec3(
			x * x / ( 2.0 * a * ( 1.0 - a ) ),
			( x - 0.5 * a ) / ( 1.0 - a ),
			1.0 - ( ( 1.0 - x ) * ( 1.0 - x ) / ( 2.0 * a * ( 1.0 - a ) ) )
		);
		float threshold = ( x < ( 1.0 - a ) )
			? ( ( x < a ) ? cases.x : cases.y )
			: cases.z;
		return clamp( threshold , 1.0e-6, 1.0 );
	}
#endif`,Cf=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,Pf=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,Df=`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,Lf=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,Uf=`#ifdef USE_AOMAP
	float ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;
	reflectedLight.indirectDiffuse *= ambientOcclusion;
	#if defined( USE_CLEARCOAT ) 
		clearcoatSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_SHEEN ) 
		sheenSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD )
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
	#endif
#endif`,If=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,Ff=`#ifdef USE_BATCHING
	#if ! defined( GL_ANGLE_multi_draw )
	#define gl_DrawID _gl_DrawID
	uniform int _gl_DrawID;
	#endif
	uniform highp sampler2D batchingTexture;
	uniform highp usampler2D batchingIdTexture;
	mat4 getBatchingMatrix( const in float i ) {
		int size = textureSize( batchingTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( batchingTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( batchingTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( batchingTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( batchingTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
	float getIndirectIndex( const in int i ) {
		int size = textureSize( batchingIdTexture, 0 ).x;
		int x = i % size;
		int y = i / size;
		return float( texelFetch( batchingIdTexture, ivec2( x, y ), 0 ).r );
	}
#endif
#ifdef USE_BATCHING_COLOR
	uniform sampler2D batchingColorTexture;
	vec3 getBatchingColor( const in float i ) {
		int size = textureSize( batchingColorTexture, 0 ).x;
		int j = int( i );
		int x = j % size;
		int y = j / size;
		return texelFetch( batchingColorTexture, ivec2( x, y ), 0 ).rgb;
	}
#endif`,Nf=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,Of=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,Bf=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,Vf=`float G_BlinnPhong_Implicit( ) {
	return 0.25;
}
float D_BlinnPhong( const in float shininess, const in float dotNH ) {
	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );
}
vec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( specularColor, 1.0, dotVH );
	float G = G_BlinnPhong_Implicit( );
	float D = D_BlinnPhong( shininess, dotNH );
	return F * ( G * D );
} // validated`,zf=`#ifdef USE_IRIDESCENCE
	const mat3 XYZ_TO_REC709 = mat3(
		 3.2404542, -0.9692660,  0.0556434,
		-1.5371385,  1.8760108, -0.2040259,
		-0.4985314,  0.0415560,  1.0572252
	);
	vec3 Fresnel0ToIor( vec3 fresnel0 ) {
		vec3 sqrtF0 = sqrt( fresnel0 );
		return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );
	}
	vec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );
	}
	float IorToFresnel0( float transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));
	}
	vec3 evalSensitivity( float OPD, vec3 shift ) {
		float phase = 2.0 * PI * OPD * 1.0e-9;
		vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
		vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
		vec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );
		vec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );
		xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );
		xyz /= 1.0685e-7;
		vec3 rgb = XYZ_TO_REC709 * xyz;
		return rgb;
	}
	vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {
		vec3 I;
		float iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );
		float sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );
		float cosTheta2Sq = 1.0 - sinTheta2Sq;
		if ( cosTheta2Sq < 0.0 ) {
			return vec3( 1.0 );
		}
		float cosTheta2 = sqrt( cosTheta2Sq );
		float R0 = IorToFresnel0( iridescenceIOR, outsideIOR );
		float R12 = F_Schlick( R0, 1.0, cosTheta1 );
		float T121 = 1.0 - R12;
		float phi12 = 0.0;
		if ( iridescenceIOR < outsideIOR ) phi12 = PI;
		float phi21 = PI - phi12;
		vec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );		vec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );
		vec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );
		vec3 phi23 = vec3( 0.0 );
		if ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;
		if ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;
		if ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;
		float OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;
		vec3 phi = vec3( phi21 ) + phi23;
		vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );
		vec3 r123 = sqrt( R123 );
		vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );
		vec3 C0 = R12 + Rs;
		I = C0;
		vec3 Cm = Rs - T121;
		for ( int m = 1; m <= 2; ++ m ) {
			Cm *= r123;
			vec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );
			I += Cm * Sm;
		}
		return max( I, vec3( 0.0 ) );
	}
#endif`,Gf=`#ifdef USE_BUMPMAP
	uniform sampler2D bumpMap;
	uniform float bumpScale;
	vec2 dHdxy_fwd() {
		vec2 dSTdx = dFdx( vBumpMapUv );
		vec2 dSTdy = dFdy( vBumpMapUv );
		float Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;
		float dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;
		float dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;
		return vec2( dBx, dBy );
	}
	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {
		vec3 vSigmaX = normalize( dFdx( surf_pos.xyz ) );
		vec3 vSigmaY = normalize( dFdy( surf_pos.xyz ) );
		vec3 vN = surf_norm;
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );
		float fDet = dot( vSigmaX, R1 ) * faceDirection;
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
	}
#endif`,Hf=`#if NUM_CLIPPING_PLANES > 0
	vec4 plane;
	#ifdef ALPHA_TO_COVERAGE
		float distanceToPlane, distanceGradient;
		float clipOpacity = 1.0;
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
			distanceGradient = fwidth( distanceToPlane ) / 2.0;
			clipOpacity *= smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			if ( clipOpacity == 0.0 ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			float unionClipOpacity = 1.0;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
				distanceGradient = fwidth( distanceToPlane ) / 2.0;
				unionClipOpacity *= 1.0 - smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			}
			#pragma unroll_loop_end
			clipOpacity *= 1.0 - unionClipOpacity;
		#endif
		diffuseColor.a *= clipOpacity;
		if ( diffuseColor.a == 0.0 ) discard;
	#else
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			if ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			bool clipped = true;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				clipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;
			}
			#pragma unroll_loop_end
			if ( clipped ) discard;
		#endif
	#endif
#endif`,kf=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,Wf=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,Xf=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,$f=`#if defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#elif defined( USE_COLOR )
	diffuseColor.rgb *= vColor;
#endif`,qf=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR )
	varying vec3 vColor;
#endif`,Yf=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec3 vColor;
#endif`,Kf=`#if defined( USE_COLOR_ALPHA )
	vColor = vec4( 1.0 );
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	vColor = vec3( 1.0 );
#endif
#ifdef USE_COLOR
	vColor *= color;
#endif
#ifdef USE_INSTANCING_COLOR
	vColor.xyz *= instanceColor.xyz;
#endif
#ifdef USE_BATCHING_COLOR
	vec3 batchingColor = getBatchingColor( getIndirectIndex( gl_DrawID ) );
	vColor.xyz *= batchingColor.xyz;
#endif`,jf=`#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract( sin( sn ) * c );
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
#ifdef USE_ALPHAHASH
	varying vec3 vPosition;
#endif
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
vec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}
float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated`,Zf=`#ifdef ENVMAP_TYPE_CUBE_UV
	#define cubeUV_minMipLevel 4.0
	#define cubeUV_minTileSize 16.0
	float getFace( vec3 direction ) {
		vec3 absDirection = abs( direction );
		float face = - 1.0;
		if ( absDirection.x > absDirection.z ) {
			if ( absDirection.x > absDirection.y )
				face = direction.x > 0.0 ? 0.0 : 3.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		} else {
			if ( absDirection.z > absDirection.y )
				face = direction.z > 0.0 ? 2.0 : 5.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		}
		return face;
	}
	vec2 getUV( vec3 direction, float face ) {
		vec2 uv;
		if ( face == 0.0 ) {
			uv = vec2( direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 1.0 ) {
			uv = vec2( - direction.x, - direction.z ) / abs( direction.y );
		} else if ( face == 2.0 ) {
			uv = vec2( - direction.x, direction.y ) / abs( direction.z );
		} else if ( face == 3.0 ) {
			uv = vec2( - direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 4.0 ) {
			uv = vec2( - direction.x, direction.z ) / abs( direction.y );
		} else {
			uv = vec2( direction.x, direction.y ) / abs( direction.z );
		}
		return 0.5 * ( uv + 1.0 );
	}
	vec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {
		float face = getFace( direction );
		float filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );
		mipInt = max( mipInt, cubeUV_minMipLevel );
		float faceSize = exp2( mipInt );
		highp vec2 uv = getUV( direction, face ) * ( faceSize - 2.0 ) + 1.0;
		if ( face > 2.0 ) {
			uv.y += faceSize;
			face -= 3.0;
		}
		uv.x += face * faceSize;
		uv.x += filterInt * 3.0 * cubeUV_minTileSize;
		uv.y += 4.0 * ( exp2( CUBEUV_MAX_MIP ) - faceSize );
		uv.x *= CUBEUV_TEXEL_WIDTH;
		uv.y *= CUBEUV_TEXEL_HEIGHT;
		#ifdef texture2DGradEXT
			return texture2DGradEXT( envMap, uv, vec2( 0.0 ), vec2( 0.0 ) ).rgb;
		#else
			return texture2D( envMap, uv ).rgb;
		#endif
	}
	#define cubeUV_r0 1.0
	#define cubeUV_m0 - 2.0
	#define cubeUV_r1 0.8
	#define cubeUV_m1 - 1.0
	#define cubeUV_r4 0.4
	#define cubeUV_m4 2.0
	#define cubeUV_r5 0.305
	#define cubeUV_m5 3.0
	#define cubeUV_r6 0.21
	#define cubeUV_m6 4.0
	float roughnessToMip( float roughness ) {
		float mip = 0.0;
		if ( roughness >= cubeUV_r1 ) {
			mip = ( cubeUV_r0 - roughness ) * ( cubeUV_m1 - cubeUV_m0 ) / ( cubeUV_r0 - cubeUV_r1 ) + cubeUV_m0;
		} else if ( roughness >= cubeUV_r4 ) {
			mip = ( cubeUV_r1 - roughness ) * ( cubeUV_m4 - cubeUV_m1 ) / ( cubeUV_r1 - cubeUV_r4 ) + cubeUV_m1;
		} else if ( roughness >= cubeUV_r5 ) {
			mip = ( cubeUV_r4 - roughness ) * ( cubeUV_m5 - cubeUV_m4 ) / ( cubeUV_r4 - cubeUV_r5 ) + cubeUV_m4;
		} else if ( roughness >= cubeUV_r6 ) {
			mip = ( cubeUV_r5 - roughness ) * ( cubeUV_m6 - cubeUV_m5 ) / ( cubeUV_r5 - cubeUV_r6 ) + cubeUV_m5;
		} else {
			mip = - 2.0 * log2( 1.16 * roughness );		}
		return mip;
	}
	vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {
		float mip = clamp( roughnessToMip( roughness ), cubeUV_m0, CUBEUV_MAX_MIP );
		float mipF = fract( mip );
		float mipInt = floor( mip );
		vec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );
		if ( mipF == 0.0 ) {
			return vec4( color0, 1.0 );
		} else {
			vec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );
			return vec4( mix( color0, color1, mipF ), 1.0 );
		}
	}
#endif`,Jf=`vec3 transformedNormal = objectNormal;
#ifdef USE_TANGENT
	vec3 transformedTangent = objectTangent;
#endif
#ifdef USE_BATCHING
	mat3 bm = mat3( batchingMatrix );
	transformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );
	transformedNormal = bm * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = bm * transformedTangent;
	#endif
#endif
#ifdef USE_INSTANCING
	mat3 im = mat3( instanceMatrix );
	transformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );
	transformedNormal = im * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = im * transformedTangent;
	#endif
#endif
transformedNormal = normalMatrix * transformedNormal;
#ifdef FLIP_SIDED
	transformedNormal = - transformedNormal;
#endif
#ifdef USE_TANGENT
	transformedTangent = ( modelViewMatrix * vec4( transformedTangent, 0.0 ) ).xyz;
	#ifdef FLIP_SIDED
		transformedTangent = - transformedTangent;
	#endif
#endif`,Qf=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,eh=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,th=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,nh=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,ih="gl_FragColor = linearToOutputTexel( gl_FragColor );",rh=`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,sh=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vec3 cameraToFrag;
		if ( isOrthographic ) {
			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToFrag = normalize( vWorldPosition - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vec3 reflectVec = reflect( cameraToFrag, worldNormal );
		#else
			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );
		#endif
	#else
		vec3 reflectVec = vReflect;
	#endif
	#ifdef ENVMAP_TYPE_CUBE
		vec4 envColor = textureCube( envMap, envMapRotation * vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );
	#else
		vec4 envColor = vec4( 0.0 );
	#endif
	#ifdef ENVMAP_BLENDING_MULTIPLY
		outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_MIX )
		outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_ADD )
		outgoingLight += envColor.xyz * specularStrength * reflectivity;
	#endif
#endif`,ah=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform float flipEnvMap;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
#endif`,oh=`#ifdef USE_ENVMAP
	uniform float reflectivity;
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		varying vec3 vWorldPosition;
		uniform float refractionRatio;
	#else
		varying vec3 vReflect;
	#endif
#endif`,lh=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,ch=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vWorldPosition = worldPosition.xyz;
	#else
		vec3 cameraToVertex;
		if ( isOrthographic ) {
			cameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToVertex = normalize( worldPosition.xyz - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vReflect = reflect( cameraToVertex, worldNormal );
		#else
			vReflect = refract( cameraToVertex, worldNormal, refractionRatio );
		#endif
	#endif
#endif`,uh=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,fh=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,hh=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,dh=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,ph=`#ifdef USE_GRADIENTMAP
	uniform sampler2D gradientMap;
#endif
vec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {
	float dotNL = dot( normal, lightDirection );
	vec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );
	#ifdef USE_GRADIENTMAP
		return vec3( texture2D( gradientMap, coord ).r );
	#else
		vec2 fw = fwidth( coord ) * 0.5;
		return mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );
	#endif
}`,mh=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,_h=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,gh=`varying vec3 vViewPosition;
struct LambertMaterial {
	vec3 diffuseColor;
	float specularStrength;
};
void RE_Direct_Lambert( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Lambert
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,vh=`uniform bool receiveShadow;
uniform vec3 ambientLightColor;
#if defined( USE_LIGHT_PROBES )
	uniform vec3 lightProbe[ 9 ];
#endif
vec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {
	float x = normal.x, y = normal.y, z = normal.z;
	vec3 result = shCoefficients[ 0 ] * 0.886227;
	result += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;
	result += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;
	result += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;
	result += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;
	result += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;
	result += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );
	result += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;
	result += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );
	return result;
}
vec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {
	vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
	vec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );
	return irradiance;
}
vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {
	vec3 irradiance = ambientLightColor;
	return irradiance;
}
float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {
	float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
	if ( cutoffDistance > 0.0 ) {
		distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
	}
	return distanceFalloff;
}
float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {
	return smoothstep( coneCosine, penumbraCosine, angleCosine );
}
#if NUM_DIR_LIGHTS > 0
	struct DirectionalLight {
		vec3 direction;
		vec3 color;
	};
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
	void getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {
		light.color = directionalLight.color;
		light.direction = directionalLight.direction;
		light.visible = true;
	}
#endif
#if NUM_POINT_LIGHTS > 0
	struct PointLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;
	};
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
	void getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = pointLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float lightDistance = length( lVector );
		light.color = pointLight.color;
		light.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );
		light.visible = ( light.color != vec3( 0.0 ) );
	}
#endif
#if NUM_SPOT_LIGHTS > 0
	struct SpotLight {
		vec3 position;
		vec3 direction;
		vec3 color;
		float distance;
		float decay;
		float coneCos;
		float penumbraCos;
	};
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
	void getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = spotLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float angleCos = dot( light.direction, spotLight.direction );
		float spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );
		if ( spotAttenuation > 0.0 ) {
			float lightDistance = length( lVector );
			light.color = spotLight.color * spotAttenuation;
			light.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );
			light.visible = ( light.color != vec3( 0.0 ) );
		} else {
			light.color = vec3( 0.0 );
			light.visible = false;
		}
	}
#endif
#if NUM_RECT_AREA_LIGHTS > 0
	struct RectAreaLight {
		vec3 color;
		vec3 position;
		vec3 halfWidth;
		vec3 halfHeight;
	};
	uniform sampler2D ltc_1;	uniform sampler2D ltc_2;
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if NUM_HEMI_LIGHTS > 0
	struct HemisphereLight {
		vec3 direction;
		vec3 skyColor;
		vec3 groundColor;
	};
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
	vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {
		float dotNL = dot( normal, hemiLight.direction );
		float hemiDiffuseWeight = 0.5 * dotNL + 0.5;
		vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );
		return irradiance;
	}
#endif`,xh=`#ifdef USE_ENVMAP
	vec3 getIBLIrradiance( const in vec3 normal ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * worldNormal, 1.0 );
			return PI * envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 reflectVec = reflect( - viewDir, normal );
			reflectVec = normalize( mix( reflectVec, normal, pow4( roughness ) ) );
			reflectVec = inverseTransformDirection( reflectVec, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * reflectVec, roughness );
			return envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	#ifdef USE_ANISOTROPY
		vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
			#ifdef ENVMAP_TYPE_CUBE_UV
				vec3 bentNormal = cross( bitangent, viewDir );
				bentNormal = normalize( cross( bentNormal, bitangent ) );
				bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
				return getIBLRadiance( viewDir, bentNormal, roughness );
			#else
				return vec3( 0.0 );
			#endif
		}
	#endif
#endif`,Mh=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,Sh=`varying vec3 vViewPosition;
struct ToonMaterial {
	vec3 diffuseColor;
};
void RE_Direct_Toon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 irradiance = getGradientIrradiance( geometryNormal, directLight.direction ) * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Toon
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,yh=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,Eh=`varying vec3 vViewPosition;
struct BlinnPhongMaterial {
	vec3 diffuseColor;
	vec3 specularColor;
	float specularShininess;
	float specularStrength;
};
void RE_Direct_BlinnPhong( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
	reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometryViewDir, geometryNormal, material.specularColor, material.specularShininess ) * material.specularStrength;
}
void RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_BlinnPhong
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,bh=`PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.diffuseContribution = diffuseColor.rgb * ( 1.0 - metalnessFactor );
material.metalness = metalnessFactor;
vec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );
material.roughness = max( roughnessFactor, 0.0525 );material.roughness += geometryRoughness;
material.roughness = min( material.roughness, 1.0 );
#ifdef IOR
	material.ior = ior;
	#ifdef USE_SPECULAR
		float specularIntensityFactor = specularIntensity;
		vec3 specularColorFactor = specularColor;
		#ifdef USE_SPECULAR_COLORMAP
			specularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;
		#endif
		#ifdef USE_SPECULAR_INTENSITYMAP
			specularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;
		#endif
		material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );
	#else
		float specularIntensityFactor = 1.0;
		vec3 specularColorFactor = vec3( 1.0 );
		material.specularF90 = 1.0;
	#endif
	material.specularColor = min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor;
	material.specularColorBlended = mix( material.specularColor, diffuseColor.rgb, metalnessFactor );
#else
	material.specularColor = vec3( 0.04 );
	material.specularColorBlended = mix( material.specularColor, diffuseColor.rgb, metalnessFactor );
	material.specularF90 = 1.0;
#endif
#ifdef USE_CLEARCOAT
	material.clearcoat = clearcoat;
	material.clearcoatRoughness = clearcoatRoughness;
	material.clearcoatF0 = vec3( 0.04 );
	material.clearcoatF90 = 1.0;
	#ifdef USE_CLEARCOATMAP
		material.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;
	#endif
	#ifdef USE_CLEARCOAT_ROUGHNESSMAP
		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;
	#endif
	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );
#endif
#ifdef USE_DISPERSION
	material.dispersion = dispersion;
#endif
#ifdef USE_IRIDESCENCE
	material.iridescence = iridescence;
	material.iridescenceIOR = iridescenceIOR;
	#ifdef USE_IRIDESCENCEMAP
		material.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;
	#endif
	#ifdef USE_IRIDESCENCE_THICKNESSMAP
		material.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;
	#else
		material.iridescenceThickness = iridescenceThicknessMaximum;
	#endif
#endif
#ifdef USE_SHEEN
	material.sheenColor = sheenColor;
	#ifdef USE_SHEEN_COLORMAP
		material.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;
	#endif
	material.sheenRoughness = clamp( sheenRoughness, 0.0001, 1.0 );
	#ifdef USE_SHEEN_ROUGHNESSMAP
		material.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;
	#endif
#endif
#ifdef USE_ANISOTROPY
	#ifdef USE_ANISOTROPYMAP
		mat2 anisotropyMat = mat2( anisotropyVector.x, anisotropyVector.y, - anisotropyVector.y, anisotropyVector.x );
		vec3 anisotropyPolar = texture2D( anisotropyMap, vAnisotropyMapUv ).rgb;
		vec2 anisotropyV = anisotropyMat * normalize( 2.0 * anisotropyPolar.rg - vec2( 1.0 ) ) * anisotropyPolar.b;
	#else
		vec2 anisotropyV = anisotropyVector;
	#endif
	material.anisotropy = length( anisotropyV );
	if( material.anisotropy == 0.0 ) {
		anisotropyV = vec2( 1.0, 0.0 );
	} else {
		anisotropyV /= material.anisotropy;
		material.anisotropy = saturate( material.anisotropy );
	}
	material.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );
	material.anisotropyT = tbn[ 0 ] * anisotropyV.x + tbn[ 1 ] * anisotropyV.y;
	material.anisotropyB = tbn[ 1 ] * anisotropyV.x - tbn[ 0 ] * anisotropyV.y;
#endif`,Th=`uniform sampler2D dfgLUT;
struct PhysicalMaterial {
	vec3 diffuseColor;
	vec3 diffuseContribution;
	vec3 specularColor;
	vec3 specularColorBlended;
	float roughness;
	float metalness;
	float specularF90;
	float dispersion;
	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif
	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0;
		vec3 iridescenceFresnelDielectric;
		vec3 iridescenceFresnelMetallic;
	#endif
	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif
	#ifdef IOR
		float ior;
	#endif
	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif
	#ifdef USE_ANISOTROPY
		float anisotropy;
		float alphaT;
		vec3 anisotropyT;
		vec3 anisotropyB;
	#endif
};
vec3 clearcoatSpecularDirect = vec3( 0.0 );
vec3 clearcoatSpecularIndirect = vec3( 0.0 );
vec3 sheenSpecularDirect = vec3( 0.0 );
vec3 sheenSpecularIndirect = vec3(0.0 );
vec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {
    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );
    float x2 = x * x;
    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );
    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );
}
float V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	return 0.5 / max( gv + gl, EPSILON );
}
float D_GGX( const in float alpha, const in float dotNH ) {
	float a2 = pow2( alpha );
	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;
	return RECIPROCAL_PI * a2 / pow2( denom );
}
#ifdef USE_ANISOTROPY
	float V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {
		float gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );
		float gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );
		float v = 0.5 / ( gv + gl );
		return v;
	}
	float D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {
		float a2 = alphaT * alphaB;
		highp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );
		highp float v2 = dot( v, v );
		float w2 = a2 / v2;
		return RECIPROCAL_PI * a2 * pow2 ( w2 );
	}
#endif
#ifdef USE_CLEARCOAT
	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {
		vec3 f0 = material.clearcoatF0;
		float f90 = material.clearcoatF90;
		float roughness = material.clearcoatRoughness;
		float alpha = pow2( roughness );
		vec3 halfDir = normalize( lightDir + viewDir );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float dotNH = saturate( dot( normal, halfDir ) );
		float dotVH = saturate( dot( viewDir, halfDir ) );
		vec3 F = F_Schlick( f0, f90, dotVH );
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
		return F * ( V * D );
	}
#endif
vec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 f0 = material.specularColorBlended;
	float f90 = material.specularF90;
	float roughness = material.roughness;
	float alpha = pow2( roughness );
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( f0, f90, dotVH );
	#ifdef USE_IRIDESCENCE
		F = mix( F, material.iridescenceFresnel, material.iridescence );
	#endif
	#ifdef USE_ANISOTROPY
		float dotTL = dot( material.anisotropyT, lightDir );
		float dotTV = dot( material.anisotropyT, viewDir );
		float dotTH = dot( material.anisotropyT, halfDir );
		float dotBL = dot( material.anisotropyB, lightDir );
		float dotBV = dot( material.anisotropyB, viewDir );
		float dotBH = dot( material.anisotropyB, halfDir );
		float V = V_GGX_SmithCorrelated_Anisotropic( material.alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );
		float D = D_GGX_Anisotropic( material.alphaT, alpha, dotNH, dotTH, dotBH );
	#else
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
	#endif
	return F * ( V * D );
}
vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;
	float dotNV = saturate( dot( N, V ) );
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
	uv = uv * LUT_SCALE + LUT_BIAS;
	return uv;
}
float LTC_ClippedSphereFormFactor( const in vec3 f ) {
	float l = length( f );
	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}
vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
	float x = dot( v1, v2 );
	float y = abs( x );
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;
	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
	return cross( v1, v2 ) * theta_sintheta;
}
vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );
	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 );
	mat3 mat = mInv * transpose( mat3( T1, T2, N ) );
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );
	return vec3( result );
}
#if defined( USE_SHEEN )
float D_Charlie( float roughness, float dotNH ) {
	float alpha = pow2( roughness );
	float invAlpha = 1.0 / alpha;
	float cos2h = dotNH * dotNH;
	float sin2h = max( 1.0 - cos2h, 0.0078125 );
	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );
}
float V_Neubelt( float dotNV, float dotNL ) {
	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );
}
vec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float D = D_Charlie( sheenRoughness, dotNH );
	float V = V_Neubelt( dotNV, dotNL );
	return sheenColor * ( D * V );
}
#endif
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	float r2 = roughness * roughness;
	float rInv = 1.0 / ( roughness + 0.1 );
	float a = -1.9362 + 1.0678 * roughness + 0.4573 * r2 - 0.8469 * rInv;
	float b = -0.6014 + 0.5538 * roughness - 0.4670 * r2 - 0.1255 * rInv;
	float DG = exp( a * dotNV + b );
	return saturate( DG );
}
vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 fab = texture2D( dfgLUT, vec2( roughness, dotNV ) ).rg;
	return specularColor * fab.x + specularF90 * fab.y;
}
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 fab = texture2D( dfgLUT, vec2( roughness, dotNV ) ).rg;
	#ifdef USE_IRIDESCENCE
		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );
	#else
		vec3 Fr = specularColor;
	#endif
	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;
	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;
	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619;	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	singleScatter += FssEss;
	multiScatter += Fms * Ems;
}
vec3 BRDF_GGX_Multiscatter( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 singleScatter = BRDF_GGX( lightDir, viewDir, normal, material );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 dfgV = texture2D( dfgLUT, vec2( material.roughness, dotNV ) ).rg;
	vec2 dfgL = texture2D( dfgLUT, vec2( material.roughness, dotNL ) ).rg;
	vec3 FssEss_V = material.specularColorBlended * dfgV.x + material.specularF90 * dfgV.y;
	vec3 FssEss_L = material.specularColorBlended * dfgL.x + material.specularF90 * dfgL.y;
	float Ess_V = dfgV.x + dfgV.y;
	float Ess_L = dfgL.x + dfgL.y;
	float Ems_V = 1.0 - Ess_V;
	float Ems_L = 1.0 - Ess_L;
	vec3 Favg = material.specularColorBlended + ( 1.0 - material.specularColorBlended ) * 0.047619;
	vec3 Fms = FssEss_V * FssEss_L * Favg / ( 1.0 - Ems_V * Ems_L * Favg + EPSILON );
	float compensationFactor = Ems_V * Ems_L;
	vec3 multiScatter = Fms * compensationFactor;
	return singleScatter + multiScatter;
}
#if NUM_RECT_AREA_LIGHTS > 0
	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
		vec3 normal = geometryNormal;
		vec3 viewDir = geometryViewDir;
		vec3 position = geometryPosition;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;
		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight;		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;
		vec2 uv = LTC_Uv( normal, viewDir, roughness );
		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );
		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);
		vec3 fresnel = ( material.specularColorBlended * t2.x + ( vec3( 1.0 ) - material.specularColorBlended ) * t2.y );
		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		reflectedLight.directDiffuse += lightColor * material.diffuseContribution * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
	}
#endif
void RE_Direct_Physical( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	#ifdef USE_CLEARCOAT
		float dotNLcc = saturate( dot( geometryClearcoatNormal, directLight.direction ) );
		vec3 ccIrradiance = dotNLcc * directLight.color;
		clearcoatSpecularDirect += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometryViewDir, geometryClearcoatNormal, material );
	#endif
	#ifdef USE_SHEEN
 
 		sheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );
 
 		float sheenAlbedoV = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
 		float sheenAlbedoL = IBLSheenBRDF( geometryNormal, directLight.direction, material.sheenRoughness );
 
 		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * max( sheenAlbedoV, sheenAlbedoL );
 
 		irradiance *= sheenEnergyComp;
 
 	#endif
	reflectedLight.directSpecular += irradiance * BRDF_GGX_Multiscatter( directLight.direction, geometryViewDir, geometryNormal, material );
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseContribution );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 diffuse = irradiance * BRDF_Lambert( material.diffuseContribution );
	#ifdef USE_SHEEN
		float sheenAlbedo = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * sheenAlbedo;
		diffuse *= sheenEnergyComp;
	#endif
	reflectedLight.indirectDiffuse += diffuse;
}
void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
	#ifdef USE_CLEARCOAT
		clearcoatSpecularIndirect += clearcoatRadiance * EnvironmentBRDF( geometryClearcoatNormal, geometryViewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularIndirect += irradiance * material.sheenColor * IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness ) * RECIPROCAL_PI;
 	#endif
	vec3 singleScatteringDielectric = vec3( 0.0 );
	vec3 multiScatteringDielectric = vec3( 0.0 );
	vec3 singleScatteringMetallic = vec3( 0.0 );
	vec3 multiScatteringMetallic = vec3( 0.0 );
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnelDielectric, material.roughness, singleScatteringDielectric, multiScatteringDielectric );
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.diffuseColor, material.specularF90, material.iridescence, material.iridescenceFresnelMetallic, material.roughness, singleScatteringMetallic, multiScatteringMetallic );
	#else
		computeMultiscattering( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.roughness, singleScatteringDielectric, multiScatteringDielectric );
		computeMultiscattering( geometryNormal, geometryViewDir, material.diffuseColor, material.specularF90, material.roughness, singleScatteringMetallic, multiScatteringMetallic );
	#endif
	vec3 singleScattering = mix( singleScatteringDielectric, singleScatteringMetallic, material.metalness );
	vec3 multiScattering = mix( multiScatteringDielectric, multiScatteringMetallic, material.metalness );
	vec3 totalScatteringDielectric = singleScatteringDielectric + multiScatteringDielectric;
	vec3 diffuse = material.diffuseContribution * ( 1.0 - totalScatteringDielectric );
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
	vec3 indirectSpecular = radiance * singleScattering;
	indirectSpecular += multiScattering * cosineWeightedIrradiance;
	vec3 indirectDiffuse = diffuse * cosineWeightedIrradiance;
	#ifdef USE_SHEEN
		float sheenAlbedo = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * sheenAlbedo;
		indirectSpecular *= sheenEnergyComp;
		indirectDiffuse *= sheenEnergyComp;
	#endif
	reflectedLight.indirectSpecular += indirectSpecular;
	reflectedLight.indirectDiffuse += indirectDiffuse;
}
#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {
	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}`,Ah=`
vec3 geometryPosition = - vViewPosition;
vec3 geometryNormal = normal;
vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
vec3 geometryClearcoatNormal = vec3( 0.0 );
#ifdef USE_CLEARCOAT
	geometryClearcoatNormal = clearcoatNormal;
#endif
#ifdef USE_IRIDESCENCE
	float dotNVi = saturate( dot( normal, geometryViewDir ) );
	if ( material.iridescenceThickness == 0.0 ) {
		material.iridescence = 0.0;
	} else {
		material.iridescence = saturate( material.iridescence );
	}
	if ( material.iridescence > 0.0 ) {
		material.iridescenceFresnelDielectric = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
		material.iridescenceFresnelMetallic = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.diffuseColor );
		material.iridescenceFresnel = mix( material.iridescenceFresnelDielectric, material.iridescenceFresnelMetallic, material.metalness );
		material.iridescenceF0 = Schlick_to_F0( material.iridescenceFresnel, 1.0, dotNVi );
	}
#endif
IncidentLight directLight;
#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )
	PointLight pointLight;
	#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
		pointLight = pointLights[ i ];
		getPointLightInfo( pointLight, geometryPosition, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS ) && ( defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_BASIC ) )
		pointLightShadow = pointLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )
	SpotLight spotLight;
	vec4 spotColor;
	vec3 spotLightCoord;
	bool inSpotLightMap;
	#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
		spotLight = spotLights[ i ];
		getSpotLightInfo( spotLight, geometryPosition, directLight );
		#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX
		#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS
		#else
		#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#endif
		#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )
			spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;
			inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );
			spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );
			directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;
		#endif
		#undef SPOT_LIGHT_MAP_INDEX
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		spotLightShadow = spotLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
	DirectionalLight directionalLight;
	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
		directionalLight = directionalLights[ i ];
		getDirectionalLightInfo( directionalLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
		directionalLightShadow = directionalLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )
	RectAreaLight rectAreaLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {
		rectAreaLight = rectAreaLights[ i ];
		RE_Direct_RectArea( rectAreaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if defined( RE_IndirectDiffuse )
	vec3 iblIrradiance = vec3( 0.0 );
	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );
	#if defined( USE_LIGHT_PROBES )
		irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );
	#endif
	#if ( NUM_HEMI_LIGHTS > 0 )
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
			irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );
		}
		#pragma unroll_loop_end
	#endif
#endif
#if defined( RE_IndirectSpecular )
	vec3 radiance = vec3( 0.0 );
	vec3 clearcoatRadiance = vec3( 0.0 );
#endif`,wh=`#if defined( RE_IndirectDiffuse )
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
		irradiance += lightMapIrradiance;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD ) && defined( ENVMAP_TYPE_CUBE_UV )
		iblIrradiance += getIBLIrradiance( geometryNormal );
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	#ifdef USE_ANISOTROPY
		radiance += getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
	#else
		radiance += getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
	#endif
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
	#endif
#endif`,Rh=`#if defined( RE_IndirectDiffuse )
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,Ch=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,Ph=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,Dh=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,Lh=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,Uh=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,Ih=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,Fh=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
	#if defined( USE_POINTS_UV )
		vec2 uv = vUv;
	#else
		vec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;
	#endif
#endif
#ifdef USE_MAP
	diffuseColor *= texture2D( map, uv );
#endif
#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, uv ).g;
#endif`,Nh=`#if defined( USE_POINTS_UV )
	varying vec2 vUv;
#else
	#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
		uniform mat3 uvTransform;
	#endif
#endif
#ifdef USE_MAP
	uniform sampler2D map;
#endif
#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,Oh=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,Bh=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,Vh=`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,zh=`#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,Gh=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,Hh=`#ifdef USE_MORPHTARGETS
	#ifndef USE_INSTANCING_MORPH
		uniform float morphTargetBaseInfluence;
		uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	#endif
	uniform sampler2DArray morphTargetsTexture;
	uniform ivec2 morphTargetsTextureSize;
	vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {
		int texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;
		int y = texelIndex / morphTargetsTextureSize.x;
		int x = texelIndex - y * morphTargetsTextureSize.x;
		ivec3 morphUV = ivec3( x, y, morphTargetIndex );
		return texelFetch( morphTargetsTexture, morphUV, 0 );
	}
#endif`,kh=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,Wh=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
#ifdef FLAT_SHADED
	vec3 fdx = dFdx( vViewPosition );
	vec3 fdy = dFdy( vViewPosition );
	vec3 normal = normalize( cross( fdx, fdy ) );
#else
	vec3 normal = normalize( vNormal );
	#ifdef DOUBLE_SIDED
		normal *= faceDirection;
	#endif
#endif
#if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )
	#ifdef USE_TANGENT
		mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn = getTangentFrame( - vViewPosition, normal,
		#if defined( USE_NORMALMAP )
			vNormalMapUv
		#elif defined( USE_CLEARCOAT_NORMALMAP )
			vClearcoatNormalMapUv
		#else
			vUv
		#endif
		);
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn[0] *= faceDirection;
		tbn[1] *= faceDirection;
	#endif
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	#ifdef USE_TANGENT
		mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn2[0] *= faceDirection;
		tbn2[1] *= faceDirection;
	#endif
#endif
vec3 nonPerturbedNormal = normal;`,Xh=`#ifdef USE_NORMALMAP_OBJECTSPACE
	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#ifdef FLIP_SIDED
		normal = - normal;
	#endif
	#ifdef DOUBLE_SIDED
		normal = normal * faceDirection;
	#endif
	normal = normalize( normalMatrix * normal );
#elif defined( USE_NORMALMAP_TANGENTSPACE )
	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	mapN.xy *= normalScale;
	normal = normalize( tbn * mapN );
#elif defined( USE_BUMPMAP )
	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
#endif`,$h=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,qh=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,Yh=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
	#endif
#endif`,Kh=`#ifdef USE_NORMALMAP
	uniform sampler2D normalMap;
	uniform vec2 normalScale;
#endif
#ifdef USE_NORMALMAP_OBJECTSPACE
	uniform mat3 normalMatrix;
#endif
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY ) )
	mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {
		vec3 q0 = dFdx( eye_pos.xyz );
		vec3 q1 = dFdy( eye_pos.xyz );
		vec2 st0 = dFdx( uv.st );
		vec2 st1 = dFdy( uv.st );
		vec3 N = surf_norm;
		vec3 q1perp = cross( q1, N );
		vec3 q0perp = cross( N, q0 );
		vec3 T = q1perp * st0.x + q0perp * st1.x;
		vec3 B = q1perp * st0.y + q0perp * st1.y;
		float det = max( dot( T, T ), dot( B, B ) );
		float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );
		return mat3( T * scale, B * scale, N );
	}
#endif`,jh=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,Zh=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,Jh=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,Qh=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,ed=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,td=`vec3 packNormalToRGB( const in vec3 normal ) {
	return normalize( normal ) * 0.5 + 0.5;
}
vec3 unpackRGBToNormal( const in vec3 rgb ) {
	return 2.0 * rgb.xyz - 1.0;
}
const float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;const float ShiftRight8 = 1. / 256.;
const float Inv255 = 1. / 255.;
const vec4 PackFactors = vec4( 1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
const vec2 UnpackFactors2 = vec2( UnpackDownscale, 1.0 / PackFactors.g );
const vec3 UnpackFactors3 = vec3( UnpackDownscale / PackFactors.rg, 1.0 / PackFactors.b );
const vec4 UnpackFactors4 = vec4( UnpackDownscale / PackFactors.rgb, 1.0 / PackFactors.a );
vec4 packDepthToRGBA( const in float v ) {
	if( v <= 0.0 )
		return vec4( 0., 0., 0., 0. );
	if( v >= 1.0 )
		return vec4( 1., 1., 1., 1. );
	float vuf;
	float af = modf( v * PackFactors.a, vuf );
	float bf = modf( vuf * ShiftRight8, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec4( vuf * Inv255, gf * PackUpscale, bf * PackUpscale, af );
}
vec3 packDepthToRGB( const in float v ) {
	if( v <= 0.0 )
		return vec3( 0., 0., 0. );
	if( v >= 1.0 )
		return vec3( 1., 1., 1. );
	float vuf;
	float bf = modf( v * PackFactors.b, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec3( vuf * Inv255, gf * PackUpscale, bf );
}
vec2 packDepthToRG( const in float v ) {
	if( v <= 0.0 )
		return vec2( 0., 0. );
	if( v >= 1.0 )
		return vec2( 1., 1. );
	float vuf;
	float gf = modf( v * 256., vuf );
	return vec2( vuf * Inv255, gf );
}
float unpackRGBAToDepth( const in vec4 v ) {
	return dot( v, UnpackFactors4 );
}
float unpackRGBToDepth( const in vec3 v ) {
	return dot( v, UnpackFactors3 );
}
float unpackRGToDepth( const in vec2 v ) {
	return v.r * UnpackFactors2.r + v.g * UnpackFactors2.g;
}
vec4 pack2HalfToRGBA( const in vec2 v ) {
	vec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );
	return vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );
}
vec2 unpackRGBATo2Half( const in vec4 v ) {
	return vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );
}
float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
	return ( viewZ + near ) / ( near - far );
}
float orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return depth * ( near - far ) - near;
}
float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
	return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );
}
float perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return ( near * far ) / ( ( far - near ) * depth - far );
}`,nd=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,id=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,rd=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,sd=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,ad=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,od=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,ld=`#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform sampler2DShadow directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		#else
			uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		#endif
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform sampler2DShadow spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		#else
			uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		#endif
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform samplerCubeShadow pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		#elif defined( SHADOWMAP_TYPE_BASIC )
			uniform samplerCube pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		#endif
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
	#if defined( SHADOWMAP_TYPE_PCF )
		float interleavedGradientNoise( vec2 position ) {
			return fract( 52.9829189 * fract( dot( position, vec2( 0.06711056, 0.00583715 ) ) ) );
		}
		vec2 vogelDiskSample( int sampleIndex, int samplesCount, float phi ) {
			const float goldenAngle = 2.399963229728653;
			float r = sqrt( ( float( sampleIndex ) + 0.5 ) / float( samplesCount ) );
			float theta = float( sampleIndex ) * goldenAngle + phi;
			return vec2( cos( theta ), sin( theta ) ) * r;
		}
	#endif
	#if defined( SHADOWMAP_TYPE_PCF )
		float getShadow( sampler2DShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			shadowCoord.z += shadowBias;
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
				float radius = shadowRadius * texelSize.x;
				float phi = interleavedGradientNoise( gl_FragCoord.xy ) * 6.28318530718;
				shadow = (
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 0, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 1, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 2, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 3, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 4, 5, phi ) * radius, shadowCoord.z ) )
				) * 0.2;
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#elif defined( SHADOWMAP_TYPE_VSM )
		float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			shadowCoord.z += shadowBias;
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				vec2 distribution = texture2D( shadowMap, shadowCoord.xy ).rg;
				float mean = distribution.x;
				float variance = distribution.y * distribution.y;
				#ifdef USE_REVERSED_DEPTH_BUFFER
					float hard_shadow = step( mean, shadowCoord.z );
				#else
					float hard_shadow = step( shadowCoord.z, mean );
				#endif
				if ( hard_shadow == 1.0 ) {
					shadow = 1.0;
				} else {
					variance = max( variance, 0.0000001 );
					float d = shadowCoord.z - mean;
					float p_max = variance / ( variance + d * d );
					p_max = clamp( ( p_max - 0.3 ) / 0.65, 0.0, 1.0 );
					shadow = max( hard_shadow, p_max );
				}
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#else
		float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			shadowCoord.z += shadowBias;
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				float depth = texture2D( shadowMap, shadowCoord.xy ).r;
				#ifdef USE_REVERSED_DEPTH_BUFFER
					shadow = step( depth, shadowCoord.z );
				#else
					shadow = step( shadowCoord.z, depth );
				#endif
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
	#if defined( SHADOWMAP_TYPE_PCF )
	float getPointShadow( samplerCubeShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		vec3 bd3D = normalize( lightToPosition );
		vec3 absVec = abs( lightToPosition );
		float viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );
		if ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {
			float dp = ( shadowCameraFar * ( viewSpaceZ - shadowCameraNear ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
			dp += shadowBias;
			float texelSize = shadowRadius / shadowMapSize.x;
			vec3 absDir = abs( bd3D );
			vec3 tangent = absDir.x > absDir.z ? vec3( 0.0, 1.0, 0.0 ) : vec3( 1.0, 0.0, 0.0 );
			tangent = normalize( cross( bd3D, tangent ) );
			vec3 bitangent = cross( bd3D, tangent );
			float phi = interleavedGradientNoise( gl_FragCoord.xy ) * 6.28318530718;
			shadow = (
				texture( shadowMap, vec4( bd3D + ( tangent * vogelDiskSample( 0, 5, phi ).x + bitangent * vogelDiskSample( 0, 5, phi ).y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * vogelDiskSample( 1, 5, phi ).x + bitangent * vogelDiskSample( 1, 5, phi ).y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * vogelDiskSample( 2, 5, phi ).x + bitangent * vogelDiskSample( 2, 5, phi ).y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * vogelDiskSample( 3, 5, phi ).x + bitangent * vogelDiskSample( 3, 5, phi ).y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * vogelDiskSample( 4, 5, phi ).x + bitangent * vogelDiskSample( 4, 5, phi ).y ) * texelSize, dp ) )
			) * 0.2;
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	#elif defined( SHADOWMAP_TYPE_BASIC )
	float getPointShadow( samplerCube shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		vec3 bd3D = normalize( lightToPosition );
		vec3 absVec = abs( lightToPosition );
		float viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );
		if ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {
			float dp = ( shadowCameraFar * ( viewSpaceZ - shadowCameraNear ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
			dp += shadowBias;
			float depth = textureCube( shadowMap, bd3D ).r;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				shadow = step( depth, dp );
			#else
				shadow = step( dp, depth );
			#endif
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	#endif
	#endif
#endif`,cd=`#if NUM_SPOT_LIGHT_COORDS > 0
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
#endif`,ud=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
	vec3 shadowWorldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
	vec4 shadowWorldPosition;
#endif
#if defined( USE_SHADOWMAP )
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );
			vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );
			vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
#endif
#if NUM_SPOT_LIGHT_COORDS > 0
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {
		shadowWorldPosition = worldPosition;
		#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
			shadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;
		#endif
		vSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;
	}
	#pragma unroll_loop_end
#endif`,fd=`float getShadowMask() {
	float shadow = 1.0;
	#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
		directionalLight = directionalLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowIntensity, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {
		spotLight = spotLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowIntensity, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0 && ( defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_BASIC ) )
	PointLightShadow pointLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
		pointLight = pointLightShadows[ i ];
		shadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowIntensity, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#endif
	return shadow;
}`,hd=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,dd=`#ifdef USE_SKINNING
	uniform mat4 bindMatrix;
	uniform mat4 bindMatrixInverse;
	uniform highp sampler2D boneTexture;
	mat4 getBoneMatrix( const in float i ) {
		int size = textureSize( boneTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
#endif`,pd=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,md=`#ifdef USE_SKINNING
	mat4 skinMatrix = mat4( 0.0 );
	skinMatrix += skinWeight.x * boneMatX;
	skinMatrix += skinWeight.y * boneMatY;
	skinMatrix += skinWeight.z * boneMatZ;
	skinMatrix += skinWeight.w * boneMatW;
	skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
	#ifdef USE_TANGENT
		objectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;
	#endif
#endif`,_d=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,gd=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,vd=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,xd=`#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
uniform float toneMappingExposure;
vec3 LinearToneMapping( vec3 color ) {
	return saturate( toneMappingExposure * color );
}
vec3 ReinhardToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	return saturate( color / ( vec3( 1.0 ) + color ) );
}
vec3 CineonToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	color = max( vec3( 0.0 ), color - 0.004 );
	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );
}
vec3 RRTAndODTFit( vec3 v ) {
	vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
	vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
	return a / b;
}
vec3 ACESFilmicToneMapping( vec3 color ) {
	const mat3 ACESInputMat = mat3(
		vec3( 0.59719, 0.07600, 0.02840 ),		vec3( 0.35458, 0.90834, 0.13383 ),
		vec3( 0.04823, 0.01566, 0.83777 )
	);
	const mat3 ACESOutputMat = mat3(
		vec3(  1.60475, -0.10208, -0.00327 ),		vec3( -0.53108,  1.10813, -0.07276 ),
		vec3( -0.07367, -0.00605,  1.07602 )
	);
	color *= toneMappingExposure / 0.6;
	color = ACESInputMat * color;
	color = RRTAndODTFit( color );
	color = ACESOutputMat * color;
	return saturate( color );
}
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
	vec3( 1.6605, - 0.1246, - 0.0182 ),
	vec3( - 0.5876, 1.1329, - 0.1006 ),
	vec3( - 0.0728, - 0.0083, 1.1187 )
);
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
	vec3( 0.6274, 0.0691, 0.0164 ),
	vec3( 0.3293, 0.9195, 0.0880 ),
	vec3( 0.0433, 0.0113, 0.8956 )
);
vec3 agxDefaultContrastApprox( vec3 x ) {
	vec3 x2 = x * x;
	vec3 x4 = x2 * x2;
	return + 15.5 * x4 * x2
		- 40.14 * x4 * x
		+ 31.96 * x4
		- 6.868 * x2 * x
		+ 0.4298 * x2
		+ 0.1191 * x
		- 0.00232;
}
vec3 AgXToneMapping( vec3 color ) {
	const mat3 AgXInsetMatrix = mat3(
		vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
		vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
		vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
	);
	const mat3 AgXOutsetMatrix = mat3(
		vec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
		vec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
		vec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
	);
	const float AgxMinEv = - 12.47393;	const float AgxMaxEv = 4.026069;
	color *= toneMappingExposure;
	color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
	color = AgXInsetMatrix * color;
	color = max( color, 1e-10 );	color = log2( color );
	color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );
	color = clamp( color, 0.0, 1.0 );
	color = agxDefaultContrastApprox( color );
	color = AgXOutsetMatrix * color;
	color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );
	color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
	color = clamp( color, 0.0, 1.0 );
	return color;
}
vec3 NeutralToneMapping( vec3 color ) {
	const float StartCompression = 0.8 - 0.04;
	const float Desaturation = 0.15;
	color *= toneMappingExposure;
	float x = min( color.r, min( color.g, color.b ) );
	float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
	color -= offset;
	float peak = max( color.r, max( color.g, color.b ) );
	if ( peak < StartCompression ) return color;
	float d = 1. - StartCompression;
	float newPeak = 1. - d * d / ( peak + d - StartCompression );
	color *= newPeak / peak;
	float g = 1. - 1. / ( Desaturation * ( peak - newPeak ) + 1. );
	return mix( color, vec3( newPeak ), g );
}
vec3 CustomToneMapping( vec3 color ) { return color; }`,Md=`#ifdef USE_TRANSMISSION
	material.transmission = transmission;
	material.transmissionAlpha = 1.0;
	material.thickness = thickness;
	material.attenuationDistance = attenuationDistance;
	material.attenuationColor = attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		material.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;
	#endif
	#ifdef USE_THICKNESSMAP
		material.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;
	#endif
	vec3 pos = vWorldPosition;
	vec3 v = normalize( cameraPosition - pos );
	vec3 n = inverseTransformDirection( normal, viewMatrix );
	vec4 transmitted = getIBLVolumeRefraction(
		n, v, material.roughness, material.diffuseContribution, material.specularColorBlended, material.specularF90,
		pos, modelMatrix, viewMatrix, projectionMatrix, material.dispersion, material.ior, material.thickness,
		material.attenuationColor, material.attenuationDistance );
	material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );
	totalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );
#endif`,Sd=`#ifdef USE_TRANSMISSION
	uniform float transmission;
	uniform float thickness;
	uniform float attenuationDistance;
	uniform vec3 attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		uniform sampler2D transmissionMap;
	#endif
	#ifdef USE_THICKNESSMAP
		uniform sampler2D thicknessMap;
	#endif
	uniform vec2 transmissionSamplerSize;
	uniform sampler2D transmissionSamplerMap;
	uniform mat4 modelMatrix;
	uniform mat4 projectionMatrix;
	varying vec3 vWorldPosition;
	float w0( float a ) {
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );
	}
	float w1( float a ) {
		return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );
	}
	float w2( float a ){
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );
	}
	float w3( float a ) {
		return ( 1.0 / 6.0 ) * ( a * a * a );
	}
	float g0( float a ) {
		return w0( a ) + w1( a );
	}
	float g1( float a ) {
		return w2( a ) + w3( a );
	}
	float h0( float a ) {
		return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );
	}
	float h1( float a ) {
		return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );
	}
	vec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, float lod ) {
		uv = uv * texelSize.zw + 0.5;
		vec2 iuv = floor( uv );
		vec2 fuv = fract( uv );
		float g0x = g0( fuv.x );
		float g1x = g1( fuv.x );
		float h0x = h0( fuv.x );
		float h1x = h1( fuv.x );
		float h0y = h0( fuv.y );
		float h1y = h1( fuv.y );
		vec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		vec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		return g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +
			g1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );
	}
	vec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {
		vec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );
		vec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );
		vec2 fLodSizeInv = 1.0 / fLodSize;
		vec2 cLodSizeInv = 1.0 / cLodSize;
		vec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), floor( lod ) );
		vec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), ceil( lod ) );
		return mix( fSample, cSample, fract( lod ) );
	}
	vec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {
		vec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );
		vec3 modelScale;
		modelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );
		modelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );
		modelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );
		return normalize( refractionVector ) * thickness * modelScale;
	}
	float applyIorToRoughness( const in float roughness, const in float ior ) {
		return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );
	}
	vec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {
		float lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );
		return textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );
	}
	vec3 volumeAttenuation( const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {
		if ( isinf( attenuationDistance ) ) {
			return vec3( 1.0 );
		} else {
			vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;
			vec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );			return transmittance;
		}
	}
	vec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,
		const in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,
		const in mat4 viewMatrix, const in mat4 projMatrix, const in float dispersion, const in float ior, const in float thickness,
		const in vec3 attenuationColor, const in float attenuationDistance ) {
		vec4 transmittedLight;
		vec3 transmittance;
		#ifdef USE_DISPERSION
			float halfSpread = ( ior - 1.0 ) * 0.025 * dispersion;
			vec3 iors = vec3( ior - halfSpread, ior, ior + halfSpread );
			for ( int i = 0; i < 3; i ++ ) {
				vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, iors[ i ], modelMatrix );
				vec3 refractedRayExit = position + transmissionRay;
				vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
				vec2 refractionCoords = ndcPos.xy / ndcPos.w;
				refractionCoords += 1.0;
				refractionCoords /= 2.0;
				vec4 transmissionSample = getTransmissionSample( refractionCoords, roughness, iors[ i ] );
				transmittedLight[ i ] = transmissionSample[ i ];
				transmittedLight.a += transmissionSample.a;
				transmittance[ i ] = diffuseColor[ i ] * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance )[ i ];
			}
			transmittedLight.a /= 3.0;
		#else
			vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );
			vec3 refractedRayExit = position + transmissionRay;
			vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
			vec2 refractionCoords = ndcPos.xy / ndcPos.w;
			refractionCoords += 1.0;
			refractionCoords /= 2.0;
			transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );
			transmittance = diffuseColor * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance );
		#endif
		vec3 attenuatedColor = transmittance * transmittedLight.rgb;
		vec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );
		float transmittanceFactor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;
		return vec4( ( 1.0 - F ) * attenuatedColor, 1.0 - ( 1.0 - transmittedLight.a ) * transmittanceFactor );
	}
#endif`,yd=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_SPECULARMAP
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,Ed=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	uniform mat3 mapTransform;
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	uniform mat3 alphaMapTransform;
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	uniform mat3 lightMapTransform;
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	uniform mat3 aoMapTransform;
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	uniform mat3 bumpMapTransform;
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	uniform mat3 normalMapTransform;
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_DISPLACEMENTMAP
	uniform mat3 displacementMapTransform;
	varying vec2 vDisplacementMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	uniform mat3 emissiveMapTransform;
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	uniform mat3 metalnessMapTransform;
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	uniform mat3 roughnessMapTransform;
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	uniform mat3 anisotropyMapTransform;
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	uniform mat3 clearcoatMapTransform;
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform mat3 clearcoatNormalMapTransform;
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform mat3 clearcoatRoughnessMapTransform;
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	uniform mat3 sheenColorMapTransform;
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	uniform mat3 sheenRoughnessMapTransform;
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	uniform mat3 iridescenceMapTransform;
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform mat3 iridescenceThicknessMapTransform;
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SPECULARMAP
	uniform mat3 specularMapTransform;
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	uniform mat3 specularColorMapTransform;
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	uniform mat3 specularIntensityMapTransform;
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,bd=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	vUv = vec3( uv, 1 ).xy;
#endif
#ifdef USE_MAP
	vMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ALPHAMAP
	vAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_LIGHTMAP
	vLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_AOMAP
	vAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_BUMPMAP
	vBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_NORMALMAP
	vNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_DISPLACEMENTMAP
	vDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_EMISSIVEMAP
	vEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_METALNESSMAP
	vMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ROUGHNESSMAP
	vRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ANISOTROPYMAP
	vAnisotropyMapUv = ( anisotropyMapTransform * vec3( ANISOTROPYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOATMAP
	vClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	vClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	vClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCEMAP
	vIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	vIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_COLORMAP
	vSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	vSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULARMAP
	vSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_COLORMAP
	vSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	vSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_TRANSMISSIONMAP
	vTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_THICKNESSMAP
	vThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;
#endif`,Td=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`;const Ad=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,wd=`uniform sampler2D t2D;
uniform float backgroundIntensity;
varying vec2 vUv;
void main() {
	vec4 texColor = texture2D( t2D, vUv );
	#ifdef DECODE_VIDEO_TEXTURE
		texColor = vec4( mix( pow( texColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), texColor.rgb * 0.0773993808, vec3( lessThanEqual( texColor.rgb, vec3( 0.04045 ) ) ) ), texColor.w );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Rd=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Cd=`#ifdef ENVMAP_TYPE_CUBE
	uniform samplerCube envMap;
#elif defined( ENVMAP_TYPE_CUBE_UV )
	uniform sampler2D envMap;
#endif
uniform float flipEnvMap;
uniform float backgroundBlurriness;
uniform float backgroundIntensity;
uniform mat3 backgroundRotation;
varying vec3 vWorldDirection;
#include <cube_uv_reflection_fragment>
void main() {
	#ifdef ENVMAP_TYPE_CUBE
		vec4 texColor = textureCube( envMap, backgroundRotation * vec3( flipEnvMap * vWorldDirection.x, vWorldDirection.yz ) );
	#elif defined( ENVMAP_TYPE_CUBE_UV )
		vec4 texColor = textureCubeUV( envMap, backgroundRotation * vWorldDirection, backgroundBlurriness );
	#else
		vec4 texColor = vec4( 0.0, 0.0, 0.0, 1.0 );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Pd=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Dd=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Ld=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
varying vec2 vHighPrecisionZW;
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vHighPrecisionZW = gl_Position.zw;
}`,Ud=`#if DEPTH_PACKING == 3200
	uniform float opacity;
#endif
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
varying vec2 vHighPrecisionZW;
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#if DEPTH_PACKING == 3200
		diffuseColor.a = opacity;
	#endif
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <logdepthbuf_fragment>
	#ifdef USE_REVERSED_DEPTH_BUFFER
		float fragCoordZ = vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ];
	#else
		float fragCoordZ = 0.5 * vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ] + 0.5;
	#endif
	#if DEPTH_PACKING == 3200
		gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );
	#elif DEPTH_PACKING == 3201
		gl_FragColor = packDepthToRGBA( fragCoordZ );
	#elif DEPTH_PACKING == 3202
		gl_FragColor = vec4( packDepthToRGB( fragCoordZ ), 1.0 );
	#elif DEPTH_PACKING == 3203
		gl_FragColor = vec4( packDepthToRG( fragCoordZ ), 0.0, 1.0 );
	#endif
}`,Id=`#define DISTANCE
varying vec3 vWorldPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <worldpos_vertex>
	#include <clipping_planes_vertex>
	vWorldPosition = worldPosition.xyz;
}`,Fd=`#define DISTANCE
uniform vec3 referencePosition;
uniform float nearDistance;
uniform float farDistance;
varying vec3 vWorldPosition;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <clipping_planes_pars_fragment>
void main () {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	float dist = length( vWorldPosition - referencePosition );
	dist = ( dist - nearDistance ) / ( farDistance - nearDistance );
	dist = saturate( dist );
	gl_FragColor = vec4( dist, 0.0, 0.0, 1.0 );
}`,Nd=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,Od=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Bd=`uniform float scale;
attribute float lineDistance;
varying float vLineDistance;
#include <common>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	vLineDistance = scale * lineDistance;
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,Vd=`uniform vec3 diffuse;
uniform float opacity;
uniform float dashSize;
uniform float totalSize;
varying float vLineDistance;
#include <common>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	if ( mod( vLineDistance, totalSize ) > dashSize ) {
		discard;
	}
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,zd=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinbase_vertex>
		#include <skinnormal_vertex>
		#include <defaultnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <fog_vertex>
}`,Gd=`uniform vec3 diffuse;
uniform float opacity;
#ifndef FLAT_SHADED
	varying vec3 vNormal;
#endif
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		reflectedLight.indirectDiffuse += lightMapTexel.rgb * lightMapIntensity * RECIPROCAL_PI;
	#else
		reflectedLight.indirectDiffuse += vec3( 1.0 );
	#endif
	#include <aomap_fragment>
	reflectedLight.indirectDiffuse *= diffuseColor.rgb;
	vec3 outgoingLight = reflectedLight.indirectDiffuse;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Hd=`#define LAMBERT
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,kd=`#define LAMBERT
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_lambert_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_lambert_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Wd=`#define MATCAP
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <displacementmap_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
	vViewPosition = - mvPosition.xyz;
}`,Xd=`#define MATCAP
uniform vec3 diffuse;
uniform float opacity;
uniform sampler2D matcap;
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	vec3 viewDir = normalize( vViewPosition );
	vec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );
	vec3 y = cross( viewDir, x );
	vec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;
	#ifdef USE_MATCAP
		vec4 matcapColor = texture2D( matcap, uv );
	#else
		vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 );
	#endif
	vec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,$d=`#define NORMAL
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	vViewPosition = - mvPosition.xyz;
#endif
}`,qd=`#define NORMAL
uniform float opacity;
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <uv_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 0.0, 0.0, 0.0, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	gl_FragColor = vec4( normalize( normal ) * 0.5 + 0.5, diffuseColor.a );
	#ifdef OPAQUE
		gl_FragColor.a = 1.0;
	#endif
}`,Yd=`#define PHONG
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Kd=`#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_phong_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,jd=`#define STANDARD
varying vec3 vViewPosition;
#ifdef USE_TRANSMISSION
	varying vec3 vWorldPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
#ifdef USE_TRANSMISSION
	vWorldPosition = worldPosition.xyz;
#endif
}`,Zd=`#define STANDARD
#ifdef PHYSICAL
	#define IOR
	#define USE_SPECULAR
#endif
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#ifdef IOR
	uniform float ior;
#endif
#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
	#ifdef USE_SPECULAR_COLORMAP
		uniform sampler2D specularColorMap;
	#endif
	#ifdef USE_SPECULAR_INTENSITYMAP
		uniform sampler2D specularIntensityMap;
	#endif
#endif
#ifdef USE_CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif
#ifdef USE_DISPERSION
	uniform float dispersion;
#endif
#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessMinimum;
	uniform float iridescenceThicknessMaximum;
#endif
#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
	#ifdef USE_SHEEN_COLORMAP
		uniform sampler2D sheenColorMap;
	#endif
	#ifdef USE_SHEEN_ROUGHNESSMAP
		uniform sampler2D sheenRoughnessMap;
	#endif
#endif
#ifdef USE_ANISOTROPY
	uniform vec2 anisotropyVector;
	#ifdef USE_ANISOTROPYMAP
		uniform sampler2D anisotropyMap;
	#endif
#endif
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <iridescence_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_physical_pars_fragment>
#include <transmission_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <iridescence_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
	#include <transmission_fragment>
	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;
	#ifdef USE_SHEEN
 
		outgoingLight = outgoingLight + sheenSpecularDirect + sheenSpecularIndirect;
 
 	#endif
	#ifdef USE_CLEARCOAT
		float dotNVcc = saturate( dot( geometryClearcoatNormal, geometryViewDir ) );
		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );
		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + ( clearcoatSpecularDirect + clearcoatSpecularIndirect ) * material.clearcoat;
	#endif
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Jd=`#define TOON
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Qd=`#define TOON
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <gradientmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_toon_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_toon_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,ep=`uniform float size;
uniform float scale;
#include <common>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
#ifdef USE_POINTS_UV
	varying vec2 vUv;
	uniform mat3 uvTransform;
#endif
void main() {
	#ifdef USE_POINTS_UV
		vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	#endif
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	gl_PointSize = size;
	#ifdef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );
	#endif
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <fog_vertex>
}`,tp=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <color_pars_fragment>
#include <map_particle_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_particle_fragment>
	#include <color_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,np=`#include <common>
#include <batching_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <shadowmap_pars_vertex>
void main() {
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,ip=`uniform vec3 color;
uniform float opacity;
#include <common>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <logdepthbuf_pars_fragment>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>
void main() {
	#include <logdepthbuf_fragment>
	gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,rp=`uniform float rotation;
uniform vec2 center;
#include <common>
#include <uv_pars_vertex>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	vec4 mvPosition = modelViewMatrix[ 3 ];
	vec2 scale = vec2( length( modelMatrix[ 0 ].xyz ), length( modelMatrix[ 1 ].xyz ) );
	#ifndef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) scale *= - mvPosition.z;
	#endif
	vec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;
	vec2 rotatedPosition;
	rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;
	rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;
	mvPosition.xy += rotatedPosition;
	gl_Position = projectionMatrix * mvPosition;
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,sp=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,ct={alphahash_fragment:wf,alphahash_pars_fragment:Rf,alphamap_fragment:Cf,alphamap_pars_fragment:Pf,alphatest_fragment:Df,alphatest_pars_fragment:Lf,aomap_fragment:Uf,aomap_pars_fragment:If,batching_pars_vertex:Ff,batching_vertex:Nf,begin_vertex:Of,beginnormal_vertex:Bf,bsdfs:Vf,iridescence_fragment:zf,bumpmap_pars_fragment:Gf,clipping_planes_fragment:Hf,clipping_planes_pars_fragment:kf,clipping_planes_pars_vertex:Wf,clipping_planes_vertex:Xf,color_fragment:$f,color_pars_fragment:qf,color_pars_vertex:Yf,color_vertex:Kf,common:jf,cube_uv_reflection_fragment:Zf,defaultnormal_vertex:Jf,displacementmap_pars_vertex:Qf,displacementmap_vertex:eh,emissivemap_fragment:th,emissivemap_pars_fragment:nh,colorspace_fragment:ih,colorspace_pars_fragment:rh,envmap_fragment:sh,envmap_common_pars_fragment:ah,envmap_pars_fragment:oh,envmap_pars_vertex:lh,envmap_physical_pars_fragment:xh,envmap_vertex:ch,fog_vertex:uh,fog_pars_vertex:fh,fog_fragment:hh,fog_pars_fragment:dh,gradientmap_pars_fragment:ph,lightmap_pars_fragment:mh,lights_lambert_fragment:_h,lights_lambert_pars_fragment:gh,lights_pars_begin:vh,lights_toon_fragment:Mh,lights_toon_pars_fragment:Sh,lights_phong_fragment:yh,lights_phong_pars_fragment:Eh,lights_physical_fragment:bh,lights_physical_pars_fragment:Th,lights_fragment_begin:Ah,lights_fragment_maps:wh,lights_fragment_end:Rh,logdepthbuf_fragment:Ch,logdepthbuf_pars_fragment:Ph,logdepthbuf_pars_vertex:Dh,logdepthbuf_vertex:Lh,map_fragment:Uh,map_pars_fragment:Ih,map_particle_fragment:Fh,map_particle_pars_fragment:Nh,metalnessmap_fragment:Oh,metalnessmap_pars_fragment:Bh,morphinstance_vertex:Vh,morphcolor_vertex:zh,morphnormal_vertex:Gh,morphtarget_pars_vertex:Hh,morphtarget_vertex:kh,normal_fragment_begin:Wh,normal_fragment_maps:Xh,normal_pars_fragment:$h,normal_pars_vertex:qh,normal_vertex:Yh,normalmap_pars_fragment:Kh,clearcoat_normal_fragment_begin:jh,clearcoat_normal_fragment_maps:Zh,clearcoat_pars_fragment:Jh,iridescence_pars_fragment:Qh,opaque_fragment:ed,packing:td,premultiplied_alpha_fragment:nd,project_vertex:id,dithering_fragment:rd,dithering_pars_fragment:sd,roughnessmap_fragment:ad,roughnessmap_pars_fragment:od,shadowmap_pars_fragment:ld,shadowmap_pars_vertex:cd,shadowmap_vertex:ud,shadowmask_pars_fragment:fd,skinbase_vertex:hd,skinning_pars_vertex:dd,skinning_vertex:pd,skinnormal_vertex:md,specularmap_fragment:_d,specularmap_pars_fragment:gd,tonemapping_fragment:vd,tonemapping_pars_fragment:xd,transmission_fragment:Md,transmission_pars_fragment:Sd,uv_pars_fragment:yd,uv_pars_vertex:Ed,uv_vertex:bd,worldpos_vertex:Td,background_vert:Ad,background_frag:wd,backgroundCube_vert:Rd,backgroundCube_frag:Cd,cube_vert:Pd,cube_frag:Dd,depth_vert:Ld,depth_frag:Ud,distance_vert:Id,distance_frag:Fd,equirect_vert:Nd,equirect_frag:Od,linedashed_vert:Bd,linedashed_frag:Vd,meshbasic_vert:zd,meshbasic_frag:Gd,meshlambert_vert:Hd,meshlambert_frag:kd,meshmatcap_vert:Wd,meshmatcap_frag:Xd,meshnormal_vert:$d,meshnormal_frag:qd,meshphong_vert:Yd,meshphong_frag:Kd,meshphysical_vert:jd,meshphysical_frag:Zd,meshtoon_vert:Jd,meshtoon_frag:Qd,points_vert:ep,points_frag:tp,shadow_vert:np,shadow_frag:ip,sprite_vert:rp,sprite_frag:sp},Pe={common:{diffuse:{value:new yt(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new st},alphaMap:{value:null},alphaMapTransform:{value:new st},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new st}},envmap:{envMap:{value:null},envMapRotation:{value:new st},flipEnvMap:{value:-1},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98},dfgLUT:{value:null}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new st}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new st}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new st},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new st},normalScale:{value:new pt(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new st},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new st}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new st}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new st}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new yt(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMap:{value:[]},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotShadowMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMap:{value:[]},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null}},points:{diffuse:{value:new yt(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new st},alphaTest:{value:0},uvTransform:{value:new st}},sprite:{diffuse:{value:new yt(16777215)},opacity:{value:1},center:{value:new pt(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new st},alphaMap:{value:null},alphaMapTransform:{value:new st},alphaTest:{value:0}}},wn={basic:{uniforms:nn([Pe.common,Pe.specularmap,Pe.envmap,Pe.aomap,Pe.lightmap,Pe.fog]),vertexShader:ct.meshbasic_vert,fragmentShader:ct.meshbasic_frag},lambert:{uniforms:nn([Pe.common,Pe.specularmap,Pe.envmap,Pe.aomap,Pe.lightmap,Pe.emissivemap,Pe.bumpmap,Pe.normalmap,Pe.displacementmap,Pe.fog,Pe.lights,{emissive:{value:new yt(0)}}]),vertexShader:ct.meshlambert_vert,fragmentShader:ct.meshlambert_frag},phong:{uniforms:nn([Pe.common,Pe.specularmap,Pe.envmap,Pe.aomap,Pe.lightmap,Pe.emissivemap,Pe.bumpmap,Pe.normalmap,Pe.displacementmap,Pe.fog,Pe.lights,{emissive:{value:new yt(0)},specular:{value:new yt(1118481)},shininess:{value:30}}]),vertexShader:ct.meshphong_vert,fragmentShader:ct.meshphong_frag},standard:{uniforms:nn([Pe.common,Pe.envmap,Pe.aomap,Pe.lightmap,Pe.emissivemap,Pe.bumpmap,Pe.normalmap,Pe.displacementmap,Pe.roughnessmap,Pe.metalnessmap,Pe.fog,Pe.lights,{emissive:{value:new yt(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:ct.meshphysical_vert,fragmentShader:ct.meshphysical_frag},toon:{uniforms:nn([Pe.common,Pe.aomap,Pe.lightmap,Pe.emissivemap,Pe.bumpmap,Pe.normalmap,Pe.displacementmap,Pe.gradientmap,Pe.fog,Pe.lights,{emissive:{value:new yt(0)}}]),vertexShader:ct.meshtoon_vert,fragmentShader:ct.meshtoon_frag},matcap:{uniforms:nn([Pe.common,Pe.bumpmap,Pe.normalmap,Pe.displacementmap,Pe.fog,{matcap:{value:null}}]),vertexShader:ct.meshmatcap_vert,fragmentShader:ct.meshmatcap_frag},points:{uniforms:nn([Pe.points,Pe.fog]),vertexShader:ct.points_vert,fragmentShader:ct.points_frag},dashed:{uniforms:nn([Pe.common,Pe.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:ct.linedashed_vert,fragmentShader:ct.linedashed_frag},depth:{uniforms:nn([Pe.common,Pe.displacementmap]),vertexShader:ct.depth_vert,fragmentShader:ct.depth_frag},normal:{uniforms:nn([Pe.common,Pe.bumpmap,Pe.normalmap,Pe.displacementmap,{opacity:{value:1}}]),vertexShader:ct.meshnormal_vert,fragmentShader:ct.meshnormal_frag},sprite:{uniforms:nn([Pe.sprite,Pe.fog]),vertexShader:ct.sprite_vert,fragmentShader:ct.sprite_frag},background:{uniforms:{uvTransform:{value:new st},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:ct.background_vert,fragmentShader:ct.background_frag},backgroundCube:{uniforms:{envMap:{value:null},flipEnvMap:{value:-1},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new st}},vertexShader:ct.backgroundCube_vert,fragmentShader:ct.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:ct.cube_vert,fragmentShader:ct.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:ct.equirect_vert,fragmentShader:ct.equirect_frag},distance:{uniforms:nn([Pe.common,Pe.displacementmap,{referencePosition:{value:new $},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:ct.distance_vert,fragmentShader:ct.distance_frag},shadow:{uniforms:nn([Pe.lights,Pe.fog,{color:{value:new yt(0)},opacity:{value:1}}]),vertexShader:ct.shadow_vert,fragmentShader:ct.shadow_frag}};wn.physical={uniforms:nn([wn.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new st},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new st},clearcoatNormalScale:{value:new pt(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new st},dispersion:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new st},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new st},sheen:{value:0},sheenColor:{value:new yt(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new st},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new st},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new st},transmissionSamplerSize:{value:new pt},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new st},attenuationDistance:{value:0},attenuationColor:{value:new yt(0)},specularColor:{value:new yt(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new st},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new st},anisotropyVector:{value:new pt},anisotropyMap:{value:null},anisotropyMapTransform:{value:new st}}]),vertexShader:ct.meshphysical_vert,fragmentShader:ct.meshphysical_frag};const Yr={r:0,b:0,g:0},pi=new Un,ap=new Ot;function op(i,e,t,n,r,s,a){const o=new yt(0);let u=s===!0?0:1,l,h,p=null,m=0,v=null;function y(C){let A=C.isScene===!0?C.background:null;return A&&A.isTexture&&(A=(C.backgroundBlurriness>0?t:e).get(A)),A}function b(C){let A=!1;const P=y(C);P===null?d(o,u):P&&P.isColor&&(d(P,1),A=!0);const D=i.xr.getEnvironmentBlendMode();D==="additive"?n.buffers.color.setClear(0,0,0,1,a):D==="alpha-blend"&&n.buffers.color.setClear(0,0,0,0,a),(i.autoClear||A)&&(n.buffers.depth.setTest(!0),n.buffers.depth.setMask(!0),n.buffers.color.setMask(!0),i.clear(i.autoClearColor,i.autoClearDepth,i.autoClearStencil))}function g(C,A){const P=y(A);P&&(P.isCubeTexture||P.mapping===ss)?(h===void 0&&(h=new In(new Ki(1,1,1),new Fn({name:"BackgroundCubeMaterial",uniforms:$i(wn.backgroundCube.uniforms),vertexShader:wn.backgroundCube.vertexShader,fragmentShader:wn.backgroundCube.fragmentShader,side:on,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),h.geometry.deleteAttribute("normal"),h.geometry.deleteAttribute("uv"),h.onBeforeRender=function(D,I,G){this.matrixWorld.copyPosition(G.matrixWorld)},Object.defineProperty(h.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),r.update(h)),pi.copy(A.backgroundRotation),pi.x*=-1,pi.y*=-1,pi.z*=-1,P.isCubeTexture&&P.isRenderTargetTexture===!1&&(pi.y*=-1,pi.z*=-1),h.material.uniforms.envMap.value=P,h.material.uniforms.flipEnvMap.value=P.isCubeTexture&&P.isRenderTargetTexture===!1?-1:1,h.material.uniforms.backgroundBlurriness.value=A.backgroundBlurriness,h.material.uniforms.backgroundIntensity.value=A.backgroundIntensity,h.material.uniforms.backgroundRotation.value.setFromMatrix4(ap.makeRotationFromEuler(pi)),h.material.toneMapped=_t.getTransfer(P.colorSpace)!==wt,(p!==P||m!==P.version||v!==i.toneMapping)&&(h.material.needsUpdate=!0,p=P,m=P.version,v=i.toneMapping),h.layers.enableAll(),C.unshift(h,h.geometry,h.material,0,0,null)):P&&P.isTexture&&(l===void 0&&(l=new In(new as(2,2),new Fn({name:"BackgroundMaterial",uniforms:$i(wn.background.uniforms),vertexShader:wn.background.vertexShader,fragmentShader:wn.background.fragmentShader,side:ai,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),l.geometry.deleteAttribute("normal"),Object.defineProperty(l.material,"map",{get:function(){return this.uniforms.t2D.value}}),r.update(l)),l.material.uniforms.t2D.value=P,l.material.uniforms.backgroundIntensity.value=A.backgroundIntensity,l.material.toneMapped=_t.getTransfer(P.colorSpace)!==wt,P.matrixAutoUpdate===!0&&P.updateMatrix(),l.material.uniforms.uvTransform.value.copy(P.matrix),(p!==P||m!==P.version||v!==i.toneMapping)&&(l.material.needsUpdate=!0,p=P,m=P.version,v=i.toneMapping),l.layers.enableAll(),C.unshift(l,l.geometry,l.material,0,0,null))}function d(C,A){C.getRGB(Yr,Wl(i)),n.buffers.color.setClear(Yr.r,Yr.g,Yr.b,A,a)}function R(){h!==void 0&&(h.geometry.dispose(),h.material.dispose(),h=void 0),l!==void 0&&(l.geometry.dispose(),l.material.dispose(),l=void 0)}return{getClearColor:function(){return o},setClearColor:function(C,A=1){o.set(C),u=A,d(o,u)},getClearAlpha:function(){return u},setClearAlpha:function(C){u=C,d(o,u)},render:b,addToRenderList:g,dispose:R}}function lp(i,e){const t=i.getParameter(i.MAX_VERTEX_ATTRIBS),n={},r=m(null);let s=r,a=!1;function o(E,B,j,K,ie){let oe=!1;const Q=p(K,j,B);s!==Q&&(s=Q,l(s.object)),oe=v(E,K,j,ie),oe&&y(E,K,j,ie),ie!==null&&e.update(ie,i.ELEMENT_ARRAY_BUFFER),(oe||a)&&(a=!1,A(E,B,j,K),ie!==null&&i.bindBuffer(i.ELEMENT_ARRAY_BUFFER,e.get(ie).buffer))}function u(){return i.createVertexArray()}function l(E){return i.bindVertexArray(E)}function h(E){return i.deleteVertexArray(E)}function p(E,B,j){const K=j.wireframe===!0;let ie=n[E.id];ie===void 0&&(ie={},n[E.id]=ie);let oe=ie[B.id];oe===void 0&&(oe={},ie[B.id]=oe);let Q=oe[K];return Q===void 0&&(Q=m(u()),oe[K]=Q),Q}function m(E){const B=[],j=[],K=[];for(let ie=0;ie<t;ie++)B[ie]=0,j[ie]=0,K[ie]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:B,enabledAttributes:j,attributeDivisors:K,object:E,attributes:{},index:null}}function v(E,B,j,K){const ie=s.attributes,oe=B.attributes;let Q=0;const te=j.getAttributes();for(const ue in te)if(te[ue].location>=0){const Ee=ie[ue];let Le=oe[ue];if(Le===void 0&&(ue==="instanceMatrix"&&E.instanceMatrix&&(Le=E.instanceMatrix),ue==="instanceColor"&&E.instanceColor&&(Le=E.instanceColor)),Ee===void 0||Ee.attribute!==Le||Le&&Ee.data!==Le.data)return!0;Q++}return s.attributesNum!==Q||s.index!==K}function y(E,B,j,K){const ie={},oe=B.attributes;let Q=0;const te=j.getAttributes();for(const ue in te)if(te[ue].location>=0){let Ee=oe[ue];Ee===void 0&&(ue==="instanceMatrix"&&E.instanceMatrix&&(Ee=E.instanceMatrix),ue==="instanceColor"&&E.instanceColor&&(Ee=E.instanceColor));const Le={};Le.attribute=Ee,Ee&&Ee.data&&(Le.data=Ee.data),ie[ue]=Le,Q++}s.attributes=ie,s.attributesNum=Q,s.index=K}function b(){const E=s.newAttributes;for(let B=0,j=E.length;B<j;B++)E[B]=0}function g(E){d(E,0)}function d(E,B){const j=s.newAttributes,K=s.enabledAttributes,ie=s.attributeDivisors;j[E]=1,K[E]===0&&(i.enableVertexAttribArray(E),K[E]=1),ie[E]!==B&&(i.vertexAttribDivisor(E,B),ie[E]=B)}function R(){const E=s.newAttributes,B=s.enabledAttributes;for(let j=0,K=B.length;j<K;j++)B[j]!==E[j]&&(i.disableVertexAttribArray(j),B[j]=0)}function C(E,B,j,K,ie,oe,Q){Q===!0?i.vertexAttribIPointer(E,B,j,ie,oe):i.vertexAttribPointer(E,B,j,K,ie,oe)}function A(E,B,j,K){b();const ie=K.attributes,oe=j.getAttributes(),Q=B.defaultAttributeValues;for(const te in oe){const ue=oe[te];if(ue.location>=0){let Ae=ie[te];if(Ae===void 0&&(te==="instanceMatrix"&&E.instanceMatrix&&(Ae=E.instanceMatrix),te==="instanceColor"&&E.instanceColor&&(Ae=E.instanceColor)),Ae!==void 0){const Ee=Ae.normalized,Le=Ae.itemSize,at=e.get(Ae);if(at===void 0)continue;const it=at.buffer,Ut=at.type,Dt=at.bytesPerElement,re=Ut===i.INT||Ut===i.UNSIGNED_INT||Ae.gpuType===Ka;if(Ae.isInterleavedBufferAttribute){const fe=Ae.data,Ne=fe.stride,Qe=Ae.offset;if(fe.isInstancedInterleavedBuffer){for(let Ve=0;Ve<ue.locationSize;Ve++)d(ue.location+Ve,fe.meshPerAttribute);E.isInstancedMesh!==!0&&K._maxInstanceCount===void 0&&(K._maxInstanceCount=fe.meshPerAttribute*fe.count)}else for(let Ve=0;Ve<ue.locationSize;Ve++)g(ue.location+Ve);i.bindBuffer(i.ARRAY_BUFFER,it);for(let Ve=0;Ve<ue.locationSize;Ve++)C(ue.location+Ve,Le/ue.locationSize,Ut,Ee,Ne*Dt,(Qe+Le/ue.locationSize*Ve)*Dt,re)}else{if(Ae.isInstancedBufferAttribute){for(let fe=0;fe<ue.locationSize;fe++)d(ue.location+fe,Ae.meshPerAttribute);E.isInstancedMesh!==!0&&K._maxInstanceCount===void 0&&(K._maxInstanceCount=Ae.meshPerAttribute*Ae.count)}else for(let fe=0;fe<ue.locationSize;fe++)g(ue.location+fe);i.bindBuffer(i.ARRAY_BUFFER,it);for(let fe=0;fe<ue.locationSize;fe++)C(ue.location+fe,Le/ue.locationSize,Ut,Ee,Le*Dt,Le/ue.locationSize*fe*Dt,re)}}else if(Q!==void 0){const Ee=Q[te];if(Ee!==void 0)switch(Ee.length){case 2:i.vertexAttrib2fv(ue.location,Ee);break;case 3:i.vertexAttrib3fv(ue.location,Ee);break;case 4:i.vertexAttrib4fv(ue.location,Ee);break;default:i.vertexAttrib1fv(ue.location,Ee)}}}}R()}function P(){G();for(const E in n){const B=n[E];for(const j in B){const K=B[j];for(const ie in K)h(K[ie].object),delete K[ie];delete B[j]}delete n[E]}}function D(E){if(n[E.id]===void 0)return;const B=n[E.id];for(const j in B){const K=B[j];for(const ie in K)h(K[ie].object),delete K[ie];delete B[j]}delete n[E.id]}function I(E){for(const B in n){const j=n[B];if(j[E.id]===void 0)continue;const K=j[E.id];for(const ie in K)h(K[ie].object),delete K[ie];delete j[E.id]}}function G(){x(),a=!0,s!==r&&(s=r,l(s.object))}function x(){r.geometry=null,r.program=null,r.wireframe=!1}return{setup:o,reset:G,resetDefaultState:x,dispose:P,releaseStatesOfGeometry:D,releaseStatesOfProgram:I,initAttributes:b,enableAttribute:g,disableUnusedAttributes:R}}function cp(i,e,t){let n;function r(l){n=l}function s(l,h){i.drawArrays(n,l,h),t.update(h,n,1)}function a(l,h,p){p!==0&&(i.drawArraysInstanced(n,l,h,p),t.update(h,n,p))}function o(l,h,p){if(p===0)return;e.get("WEBGL_multi_draw").multiDrawArraysWEBGL(n,l,0,h,0,p);let v=0;for(let y=0;y<p;y++)v+=h[y];t.update(v,n,1)}function u(l,h,p,m){if(p===0)return;const v=e.get("WEBGL_multi_draw");if(v===null)for(let y=0;y<l.length;y++)a(l[y],h[y],m[y]);else{v.multiDrawArraysInstancedWEBGL(n,l,0,h,0,m,0,p);let y=0;for(let b=0;b<p;b++)y+=h[b]*m[b];t.update(y,n,1)}}this.setMode=r,this.render=s,this.renderInstances=a,this.renderMultiDraw=o,this.renderMultiDrawInstances=u}function up(i,e,t,n){let r;function s(){if(r!==void 0)return r;if(e.has("EXT_texture_filter_anisotropic")===!0){const I=e.get("EXT_texture_filter_anisotropic");r=i.getParameter(I.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else r=0;return r}function a(I){return!(I!==En&&n.convert(I)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_FORMAT))}function o(I){const G=I===Yn&&(e.has("EXT_color_buffer_half_float")||e.has("EXT_color_buffer_float"));return!(I!==dn&&n.convert(I)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_TYPE)&&I!==Rn&&!G)}function u(I){if(I==="highp"){if(i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.HIGH_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.HIGH_FLOAT).precision>0)return"highp";I="mediump"}return I==="mediump"&&i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.MEDIUM_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}let l=t.precision!==void 0?t.precision:"highp";const h=u(l);h!==l&&(et("WebGLRenderer:",l,"not supported, using",h,"instead."),l=h);const p=t.logarithmicDepthBuffer===!0,m=t.reversedDepthBuffer===!0&&e.has("EXT_clip_control"),v=i.getParameter(i.MAX_TEXTURE_IMAGE_UNITS),y=i.getParameter(i.MAX_VERTEX_TEXTURE_IMAGE_UNITS),b=i.getParameter(i.MAX_TEXTURE_SIZE),g=i.getParameter(i.MAX_CUBE_MAP_TEXTURE_SIZE),d=i.getParameter(i.MAX_VERTEX_ATTRIBS),R=i.getParameter(i.MAX_VERTEX_UNIFORM_VECTORS),C=i.getParameter(i.MAX_VARYING_VECTORS),A=i.getParameter(i.MAX_FRAGMENT_UNIFORM_VECTORS),P=i.getParameter(i.MAX_SAMPLES),D=i.getParameter(i.SAMPLES);return{isWebGL2:!0,getMaxAnisotropy:s,getMaxPrecision:u,textureFormatReadable:a,textureTypeReadable:o,precision:l,logarithmicDepthBuffer:p,reversedDepthBuffer:m,maxTextures:v,maxVertexTextures:y,maxTextureSize:b,maxCubemapSize:g,maxAttributes:d,maxVertexUniforms:R,maxVaryings:C,maxFragmentUniforms:A,maxSamples:P,samples:D}}function fp(i){const e=this;let t=null,n=0,r=!1,s=!1;const a=new _i,o=new st,u={value:null,needsUpdate:!1};this.uniform=u,this.numPlanes=0,this.numIntersection=0,this.init=function(p,m){const v=p.length!==0||m||n!==0||r;return r=m,n=p.length,v},this.beginShadows=function(){s=!0,h(null)},this.endShadows=function(){s=!1},this.setGlobalState=function(p,m){t=h(p,m,0)},this.setState=function(p,m,v){const y=p.clippingPlanes,b=p.clipIntersection,g=p.clipShadows,d=i.get(p);if(!r||y===null||y.length===0||s&&!g)s?h(null):l();else{const R=s?0:n,C=R*4;let A=d.clippingState||null;u.value=A,A=h(y,m,C,v);for(let P=0;P!==C;++P)A[P]=t[P];d.clippingState=A,this.numIntersection=b?this.numPlanes:0,this.numPlanes+=R}};function l(){u.value!==t&&(u.value=t,u.needsUpdate=n>0),e.numPlanes=n,e.numIntersection=0}function h(p,m,v,y){const b=p!==null?p.length:0;let g=null;if(b!==0){if(g=u.value,y!==!0||g===null){const d=v+b*4,R=m.matrixWorldInverse;o.getNormalMatrix(R),(g===null||g.length<d)&&(g=new Float32Array(d));for(let C=0,A=v;C!==b;++C,A+=4)a.copy(p[C]).applyMatrix4(R,o),a.normal.toArray(g,A),g[A+3]=a.constant}u.value=g,u.needsUpdate=!0}return e.numPlanes=b,e.numIntersection=0,g}}function hp(i){let e=new WeakMap;function t(a,o){return o===oa?a.mapping=Si:o===la&&(a.mapping=ki),a}function n(a){if(a&&a.isTexture){const o=a.mapping;if(o===oa||o===la)if(e.has(a)){const u=e.get(a).texture;return t(u,a.mapping)}else{const u=a.image;if(u&&u.height>0){const l=new ql(u.height);return l.fromEquirectangularTexture(i,a),e.set(a,l),a.addEventListener("dispose",r),t(l.texture,a.mapping)}else return null}}return a}function r(a){const o=a.target;o.removeEventListener("dispose",r);const u=e.get(o);u!==void 0&&(e.delete(o),u.dispose())}function s(){e=new WeakMap}return{get:n,dispose:s}}const si=4,Zo=[.125,.215,.35,.446,.526,.582],vi=20,dp=256,lr=new Kl,Jo=new yt;let $s=null,qs=0,Ys=0,Ks=!1;const pp=new $;class Qo{constructor(e){this._renderer=e,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._sizeLods=[],this._sigmas=[],this._lodMeshes=[],this._backgroundBox=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._blurMaterial=null,this._ggxMaterial=null}fromScene(e,t=0,n=.1,r=100,s={}){const{size:a=256,position:o=pp}=s;$s=this._renderer.getRenderTarget(),qs=this._renderer.getActiveCubeFace(),Ys=this._renderer.getActiveMipmapLevel(),Ks=this._renderer.xr.enabled,this._renderer.xr.enabled=!1,this._setSize(a);const u=this._allocateTargets();return u.depthBuffer=!0,this._sceneToCubeUV(e,n,r,u,o),t>0&&this._blur(u,0,0,t),this._applyPMREM(u),this._cleanup(u),u}fromEquirectangular(e,t=null){return this._fromTexture(e,t)}fromCubemap(e,t=null){return this._fromTexture(e,t)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=nl(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=tl(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose(),this._backgroundBox!==null&&(this._backgroundBox.geometry.dispose(),this._backgroundBox.material.dispose())}_setSize(e){this._lodMax=Math.floor(Math.log2(e)),this._cubeSize=Math.pow(2,this._lodMax)}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._ggxMaterial!==null&&this._ggxMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let e=0;e<this._lodMeshes.length;e++)this._lodMeshes[e].geometry.dispose()}_cleanup(e){this._renderer.setRenderTarget($s,qs,Ys),this._renderer.xr.enabled=Ks,e.scissorTest=!1,Vi(e,0,0,e.width,e.height)}_fromTexture(e,t){e.mapping===Si||e.mapping===ki?this._setSize(e.image.length===0?16:e.image[0].width||e.image[0].image.width):this._setSize(e.image.width/4),$s=this._renderer.getRenderTarget(),qs=this._renderer.getActiveCubeFace(),Ys=this._renderer.getActiveMipmapLevel(),Ks=this._renderer.xr.enabled,this._renderer.xr.enabled=!1;const n=t||this._allocateTargets();return this._textureToCubeUV(e,n),this._applyPMREM(n),this._cleanup(n),n}_allocateTargets(){const e=3*Math.max(this._cubeSize,112),t=4*this._cubeSize,n={magFilter:Qt,minFilter:Qt,generateMipmaps:!1,type:Yn,format:En,colorSpace:Xi,depthBuffer:!1},r=el(e,t,n);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==e||this._pingPongRenderTarget.height!==t){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=el(e,t,n);const{_lodMax:s}=this;({lodMeshes:this._lodMeshes,sizeLods:this._sizeLods,sigmas:this._sigmas}=mp(s)),this._blurMaterial=gp(s,e,t),this._ggxMaterial=_p(s,e,t)}return r}_compileMaterial(e){const t=new In(new bn,e);this._renderer.compile(t,lr)}_sceneToCubeUV(e,t,n,r,s){const u=new hn(90,1,t,n),l=[1,-1,1,1,1,1],h=[1,1,1,-1,-1,-1],p=this._renderer,m=p.autoClear,v=p.toneMapping;p.getClearColor(Jo),p.toneMapping=Pn,p.autoClear=!1,p.state.buffers.depth.getReversed()&&(p.setRenderTarget(r),p.clearDepth(),p.setRenderTarget(null)),this._backgroundBox===null&&(this._backgroundBox=new In(new Ki,new Gl({name:"PMREM.Background",side:on,depthWrite:!1,depthTest:!1})));const b=this._backgroundBox,g=b.material;let d=!1;const R=e.background;R?R.isColor&&(g.color.copy(R),e.background=null,d=!0):(g.color.copy(Jo),d=!0);for(let C=0;C<6;C++){const A=C%3;A===0?(u.up.set(0,l[C],0),u.position.set(s.x,s.y,s.z),u.lookAt(s.x+h[C],s.y,s.z)):A===1?(u.up.set(0,0,l[C]),u.position.set(s.x,s.y,s.z),u.lookAt(s.x,s.y+h[C],s.z)):(u.up.set(0,l[C],0),u.position.set(s.x,s.y,s.z),u.lookAt(s.x,s.y,s.z+h[C]));const P=this._cubeSize;Vi(r,A*P,C>2?P:0,P,P),p.setRenderTarget(r),d&&p.render(b,u),p.render(e,u)}p.toneMapping=v,p.autoClear=m,e.background=R}_textureToCubeUV(e,t){const n=this._renderer,r=e.mapping===Si||e.mapping===ki;r?(this._cubemapMaterial===null&&(this._cubemapMaterial=nl()),this._cubemapMaterial.uniforms.flipEnvMap.value=e.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=tl());const s=r?this._cubemapMaterial:this._equirectMaterial,a=this._lodMeshes[0];a.material=s;const o=s.uniforms;o.envMap.value=e;const u=this._cubeSize;Vi(t,0,0,3*u,2*u),n.setRenderTarget(t),n.render(a,lr)}_applyPMREM(e){const t=this._renderer,n=t.autoClear;t.autoClear=!1;const r=this._lodMeshes.length;for(let s=1;s<r;s++)this._applyGGXFilter(e,s-1,s);t.autoClear=n}_applyGGXFilter(e,t,n){const r=this._renderer,s=this._pingPongRenderTarget,a=this._ggxMaterial,o=this._lodMeshes[n];o.material=a;const u=a.uniforms,l=n/(this._lodMeshes.length-1),h=t/(this._lodMeshes.length-1),p=Math.sqrt(l*l-h*h),m=0+l*1.25,v=p*m,{_lodMax:y}=this,b=this._sizeLods[n],g=3*b*(n>y-si?n-y+si:0),d=4*(this._cubeSize-b);u.envMap.value=e.texture,u.roughness.value=v,u.mipInt.value=y-t,Vi(s,g,d,3*b,2*b),r.setRenderTarget(s),r.render(o,lr),u.envMap.value=s.texture,u.roughness.value=0,u.mipInt.value=y-n,Vi(e,g,d,3*b,2*b),r.setRenderTarget(e),r.render(o,lr)}_blur(e,t,n,r,s){const a=this._pingPongRenderTarget;this._halfBlur(e,a,t,n,r,"latitudinal",s),this._halfBlur(a,e,n,n,r,"longitudinal",s)}_halfBlur(e,t,n,r,s,a,o){const u=this._renderer,l=this._blurMaterial;a!=="latitudinal"&&a!=="longitudinal"&&xt("blur direction must be either latitudinal or longitudinal!");const h=3,p=this._lodMeshes[r];p.material=l;const m=l.uniforms,v=this._sizeLods[n]-1,y=isFinite(s)?Math.PI/(2*v):2*Math.PI/(2*vi-1),b=s/y,g=isFinite(s)?1+Math.floor(h*b):vi;g>vi&&et(`sigmaRadians, ${s}, is too large and will clip, as it requested ${g} samples when the maximum is set to ${vi}`);const d=[];let R=0;for(let I=0;I<vi;++I){const G=I/b,x=Math.exp(-G*G/2);d.push(x),I===0?R+=x:I<g&&(R+=2*x)}for(let I=0;I<d.length;I++)d[I]=d[I]/R;m.envMap.value=e.texture,m.samples.value=g,m.weights.value=d,m.latitudinal.value=a==="latitudinal",o&&(m.poleAxis.value=o);const{_lodMax:C}=this;m.dTheta.value=y,m.mipInt.value=C-n;const A=this._sizeLods[r],P=3*A*(r>C-si?r-C+si:0),D=4*(this._cubeSize-A);Vi(t,P,D,3*A,2*A),u.setRenderTarget(t),u.render(p,lr)}}function mp(i){const e=[],t=[],n=[];let r=i;const s=i-si+1+Zo.length;for(let a=0;a<s;a++){const o=Math.pow(2,r);e.push(o);let u=1/o;a>i-si?u=Zo[a-i+si-1]:a===0&&(u=0),t.push(u);const l=1/(o-2),h=-l,p=1+l,m=[h,h,p,h,p,p,h,h,p,p,h,p],v=6,y=6,b=3,g=2,d=1,R=new Float32Array(b*y*v),C=new Float32Array(g*y*v),A=new Float32Array(d*y*v);for(let D=0;D<v;D++){const I=D%3*2/3-1,G=D>2?0:-1,x=[I,G,0,I+2/3,G,0,I+2/3,G+1,0,I,G,0,I+2/3,G+1,0,I,G+1,0];R.set(x,b*y*D),C.set(m,g*y*D);const E=[D,D,D,D,D,D];A.set(E,d*y*D)}const P=new bn;P.setAttribute("position",new _n(R,b)),P.setAttribute("uv",new _n(C,g)),P.setAttribute("faceIndex",new _n(A,d)),n.push(new In(P,null)),r>si&&r--}return{lodMeshes:n,sizeLods:e,sigmas:t}}function el(i,e,t){const n=new Dn(i,e,t);return n.texture.mapping=ss,n.texture.name="PMREM.cubeUv",n.scissorTest=!0,n}function Vi(i,e,t,n,r){i.viewport.set(e,t,n,r),i.scissor.set(e,t,n,r)}function _p(i,e,t){return new Fn({name:"PMREMGGXConvolution",defines:{GGX_SAMPLES:dp,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${i}.0`},uniforms:{envMap:{value:null},roughness:{value:0},mipInt:{value:0}},vertexShader:os(),fragmentShader:`

			precision highp float;
			precision highp int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform float roughness;
			uniform float mipInt;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			#define PI 3.14159265359

			// Van der Corput radical inverse
			float radicalInverse_VdC(uint bits) {
				bits = (bits << 16u) | (bits >> 16u);
				bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
				bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
				bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
				bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
				return float(bits) * 2.3283064365386963e-10; // / 0x100000000
			}

			// Hammersley sequence
			vec2 hammersley(uint i, uint N) {
				return vec2(float(i) / float(N), radicalInverse_VdC(i));
			}

			// GGX VNDF importance sampling (Eric Heitz 2018)
			// "Sampling the GGX Distribution of Visible Normals"
			// https://jcgt.org/published/0007/04/01/
			vec3 importanceSampleGGX_VNDF(vec2 Xi, vec3 V, float roughness) {
				float alpha = roughness * roughness;

				// Section 3.2: Transform view direction to hemisphere configuration
				vec3 Vh = normalize(vec3(alpha * V.x, alpha * V.y, V.z));

				// Section 4.1: Orthonormal basis
				float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
				vec3 T1 = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) / sqrt(lensq) : vec3(1.0, 0.0, 0.0);
				vec3 T2 = cross(Vh, T1);

				// Section 4.2: Parameterization of projected area
				float r = sqrt(Xi.x);
				float phi = 2.0 * PI * Xi.y;
				float t1 = r * cos(phi);
				float t2 = r * sin(phi);
				float s = 0.5 * (1.0 + Vh.z);
				t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

				// Section 4.3: Reprojection onto hemisphere
				vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;

				// Section 3.4: Transform back to ellipsoid configuration
				return normalize(vec3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
			}

			void main() {
				vec3 N = normalize(vOutputDirection);
				vec3 V = N; // Assume view direction equals normal for pre-filtering

				vec3 prefilteredColor = vec3(0.0);
				float totalWeight = 0.0;

				// For very low roughness, just sample the environment directly
				if (roughness < 0.001) {
					gl_FragColor = vec4(bilinearCubeUV(envMap, N, mipInt), 1.0);
					return;
				}

				// Tangent space basis for VNDF sampling
				vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
				vec3 tangent = normalize(cross(up, N));
				vec3 bitangent = cross(N, tangent);

				for(uint i = 0u; i < uint(GGX_SAMPLES); i++) {
					vec2 Xi = hammersley(i, uint(GGX_SAMPLES));

					// For PMREM, V = N, so in tangent space V is always (0, 0, 1)
					vec3 H_tangent = importanceSampleGGX_VNDF(Xi, vec3(0.0, 0.0, 1.0), roughness);

					// Transform H back to world space
					vec3 H = normalize(tangent * H_tangent.x + bitangent * H_tangent.y + N * H_tangent.z);
					vec3 L = normalize(2.0 * dot(V, H) * H - V);

					float NdotL = max(dot(N, L), 0.0);

					if(NdotL > 0.0) {
						// Sample environment at fixed mip level
						// VNDF importance sampling handles the distribution filtering
						vec3 sampleColor = bilinearCubeUV(envMap, L, mipInt);

						// Weight by NdotL for the split-sum approximation
						// VNDF PDF naturally accounts for the visible microfacet distribution
						prefilteredColor += sampleColor * NdotL;
						totalWeight += NdotL;
					}
				}

				if (totalWeight > 0.0) {
					prefilteredColor = prefilteredColor / totalWeight;
				}

				gl_FragColor = vec4(prefilteredColor, 1.0);
			}
		`,blending:$n,depthTest:!1,depthWrite:!1})}function gp(i,e,t){const n=new Float32Array(vi),r=new $(0,1,0);return new Fn({name:"SphericalGaussianBlur",defines:{n:vi,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${i}.0`},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:n},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:r}},vertexShader:os(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform int samples;
			uniform float weights[ n ];
			uniform bool latitudinal;
			uniform float dTheta;
			uniform float mipInt;
			uniform vec3 poleAxis;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			vec3 getSample( float theta, vec3 axis ) {

				float cosTheta = cos( theta );
				// Rodrigues' axis-angle rotation
				vec3 sampleDirection = vOutputDirection * cosTheta
					+ cross( axis, vOutputDirection ) * sin( theta )
					+ axis * dot( axis, vOutputDirection ) * ( 1.0 - cosTheta );

				return bilinearCubeUV( envMap, sampleDirection, mipInt );

			}

			void main() {

				vec3 axis = latitudinal ? poleAxis : cross( poleAxis, vOutputDirection );

				if ( all( equal( axis, vec3( 0.0 ) ) ) ) {

					axis = vec3( vOutputDirection.z, 0.0, - vOutputDirection.x );

				}

				axis = normalize( axis );

				gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
				gl_FragColor.rgb += weights[ 0 ] * getSample( 0.0, axis );

				for ( int i = 1; i < n; i++ ) {

					if ( i >= samples ) {

						break;

					}

					float theta = dTheta * float( i );
					gl_FragColor.rgb += weights[ i ] * getSample( -1.0 * theta, axis );
					gl_FragColor.rgb += weights[ i ] * getSample( theta, axis );

				}

			}
		`,blending:$n,depthTest:!1,depthWrite:!1})}function tl(){return new Fn({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null}},vertexShader:os(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;

			#include <common>

			void main() {

				vec3 outputDirection = normalize( vOutputDirection );
				vec2 uv = equirectUv( outputDirection );

				gl_FragColor = vec4( texture2D ( envMap, uv ).rgb, 1.0 );

			}
		`,blending:$n,depthTest:!1,depthWrite:!1})}function nl(){return new Fn({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:os(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,blending:$n,depthTest:!1,depthWrite:!1})}function os(){return`

		precision mediump float;
		precision mediump int;

		attribute float faceIndex;

		varying vec3 vOutputDirection;

		// RH coordinate system; PMREM face-indexing convention
		vec3 getDirection( vec2 uv, float face ) {

			uv = 2.0 * uv - 1.0;

			vec3 direction = vec3( uv, 1.0 );

			if ( face == 0.0 ) {

				direction = direction.zyx; // ( 1, v, u ) pos x

			} else if ( face == 1.0 ) {

				direction = direction.xzy;
				direction.xz *= -1.0; // ( -u, 1, -v ) pos y

			} else if ( face == 2.0 ) {

				direction.x *= -1.0; // ( -u, v, 1 ) pos z

			} else if ( face == 3.0 ) {

				direction = direction.zyx;
				direction.xz *= -1.0; // ( -1, v, -u ) neg x

			} else if ( face == 4.0 ) {

				direction = direction.xzy;
				direction.xy *= -1.0; // ( -u, -1, v ) neg y

			} else if ( face == 5.0 ) {

				direction.z *= -1.0; // ( u, v, -1 ) neg z

			}

			return direction;

		}

		void main() {

			vOutputDirection = getDirection( uv, faceIndex );
			gl_Position = vec4( position, 1.0 );

		}
	`}function vp(i){let e=new WeakMap,t=null;function n(o){if(o&&o.isTexture){const u=o.mapping,l=u===oa||u===la,h=u===Si||u===ki;if(l||h){let p=e.get(o);const m=p!==void 0?p.texture.pmremVersion:0;if(o.isRenderTargetTexture&&o.pmremVersion!==m)return t===null&&(t=new Qo(i)),p=l?t.fromEquirectangular(o,p):t.fromCubemap(o,p),p.texture.pmremVersion=o.pmremVersion,e.set(o,p),p.texture;if(p!==void 0)return p.texture;{const v=o.image;return l&&v&&v.height>0||h&&v&&r(v)?(t===null&&(t=new Qo(i)),p=l?t.fromEquirectangular(o):t.fromCubemap(o),p.texture.pmremVersion=o.pmremVersion,e.set(o,p),o.addEventListener("dispose",s),p.texture):null}}}return o}function r(o){let u=0;const l=6;for(let h=0;h<l;h++)o[h]!==void 0&&u++;return u===l}function s(o){const u=o.target;u.removeEventListener("dispose",s);const l=e.get(u);l!==void 0&&(e.delete(u),l.dispose())}function a(){e=new WeakMap,t!==null&&(t.dispose(),t=null)}return{get:n,dispose:a}}function xp(i){const e={};function t(n){if(e[n]!==void 0)return e[n];const r=i.getExtension(n);return e[n]=r,r}return{has:function(n){return t(n)!==null},init:function(){t("EXT_color_buffer_float"),t("WEBGL_clip_cull_distance"),t("OES_texture_float_linear"),t("EXT_color_buffer_half_float"),t("WEBGL_multisampled_render_to_texture"),t("WEBGL_render_shared_exponent")},get:function(n){const r=t(n);return r===null&&pr("WebGLRenderer: "+n+" extension not supported."),r}}}function Mp(i,e,t,n){const r={},s=new WeakMap;function a(p){const m=p.target;m.index!==null&&e.remove(m.index);for(const y in m.attributes)e.remove(m.attributes[y]);m.removeEventListener("dispose",a),delete r[m.id];const v=s.get(m);v&&(e.remove(v),s.delete(m)),n.releaseStatesOfGeometry(m),m.isInstancedBufferGeometry===!0&&delete m._maxInstanceCount,t.memory.geometries--}function o(p,m){return r[m.id]===!0||(m.addEventListener("dispose",a),r[m.id]=!0,t.memory.geometries++),m}function u(p){const m=p.attributes;for(const v in m)e.update(m[v],i.ARRAY_BUFFER)}function l(p){const m=[],v=p.index,y=p.attributes.position;let b=0;if(v!==null){const R=v.array;b=v.version;for(let C=0,A=R.length;C<A;C+=3){const P=R[C+0],D=R[C+1],I=R[C+2];m.push(P,D,D,I,I,P)}}else if(y!==void 0){const R=y.array;b=y.version;for(let C=0,A=R.length/3-1;C<A;C+=3){const P=C+0,D=C+1,I=C+2;m.push(P,D,D,I,I,P)}}else return;const g=new(Ol(m)?kl:Hl)(m,1);g.version=b;const d=s.get(p);d&&e.remove(d),s.set(p,g)}function h(p){const m=s.get(p);if(m){const v=p.index;v!==null&&m.version<v.version&&l(p)}else l(p);return s.get(p)}return{get:o,update:u,getWireframeAttribute:h}}function Sp(i,e,t){let n;function r(m){n=m}let s,a;function o(m){s=m.type,a=m.bytesPerElement}function u(m,v){i.drawElements(n,v,s,m*a),t.update(v,n,1)}function l(m,v,y){y!==0&&(i.drawElementsInstanced(n,v,s,m*a,y),t.update(v,n,y))}function h(m,v,y){if(y===0)return;e.get("WEBGL_multi_draw").multiDrawElementsWEBGL(n,v,0,s,m,0,y);let g=0;for(let d=0;d<y;d++)g+=v[d];t.update(g,n,1)}function p(m,v,y,b){if(y===0)return;const g=e.get("WEBGL_multi_draw");if(g===null)for(let d=0;d<m.length;d++)l(m[d]/a,v[d],b[d]);else{g.multiDrawElementsInstancedWEBGL(n,v,0,s,m,0,b,0,y);let d=0;for(let R=0;R<y;R++)d+=v[R]*b[R];t.update(d,n,1)}}this.setMode=r,this.setIndex=o,this.render=u,this.renderInstances=l,this.renderMultiDraw=h,this.renderMultiDrawInstances=p}function yp(i){const e={geometries:0,textures:0},t={frame:0,calls:0,triangles:0,points:0,lines:0};function n(s,a,o){switch(t.calls++,a){case i.TRIANGLES:t.triangles+=o*(s/3);break;case i.LINES:t.lines+=o*(s/2);break;case i.LINE_STRIP:t.lines+=o*(s-1);break;case i.LINE_LOOP:t.lines+=o*s;break;case i.POINTS:t.points+=o*s;break;default:xt("WebGLInfo: Unknown draw mode:",a);break}}function r(){t.calls=0,t.triangles=0,t.points=0,t.lines=0}return{memory:e,render:t,programs:null,autoReset:!0,reset:r,update:n}}function Ep(i,e,t){const n=new WeakMap,r=new Nt;function s(a,o,u){const l=a.morphTargetInfluences,h=o.morphAttributes.position||o.morphAttributes.normal||o.morphAttributes.color,p=h!==void 0?h.length:0;let m=n.get(o);if(m===void 0||m.count!==p){let x=function(){I.dispose(),n.delete(o),o.removeEventListener("dispose",x)};m!==void 0&&m.texture.dispose();const v=o.morphAttributes.position!==void 0,y=o.morphAttributes.normal!==void 0,b=o.morphAttributes.color!==void 0,g=o.morphAttributes.position||[],d=o.morphAttributes.normal||[],R=o.morphAttributes.color||[];let C=0;v===!0&&(C=1),y===!0&&(C=2),b===!0&&(C=3);let A=o.attributes.position.count*C,P=1;A>e.maxTextureSize&&(P=Math.ceil(A/e.maxTextureSize),A=e.maxTextureSize);const D=new Float32Array(A*P*4*p),I=new Bl(D,A,P,p);I.type=Rn,I.needsUpdate=!0;const G=C*4;for(let E=0;E<p;E++){const B=g[E],j=d[E],K=R[E],ie=A*P*4*E;for(let oe=0;oe<B.count;oe++){const Q=oe*G;v===!0&&(r.fromBufferAttribute(B,oe),D[ie+Q+0]=r.x,D[ie+Q+1]=r.y,D[ie+Q+2]=r.z,D[ie+Q+3]=0),y===!0&&(r.fromBufferAttribute(j,oe),D[ie+Q+4]=r.x,D[ie+Q+5]=r.y,D[ie+Q+6]=r.z,D[ie+Q+7]=0),b===!0&&(r.fromBufferAttribute(K,oe),D[ie+Q+8]=r.x,D[ie+Q+9]=r.y,D[ie+Q+10]=r.z,D[ie+Q+11]=K.itemSize===4?r.w:1)}}m={count:p,texture:I,size:new pt(A,P)},n.set(o,m),o.addEventListener("dispose",x)}if(a.isInstancedMesh===!0&&a.morphTexture!==null)u.getUniforms().setValue(i,"morphTexture",a.morphTexture,t);else{let v=0;for(let b=0;b<l.length;b++)v+=l[b];const y=o.morphTargetsRelative?1:1-v;u.getUniforms().setValue(i,"morphTargetBaseInfluence",y),u.getUniforms().setValue(i,"morphTargetInfluences",l)}u.getUniforms().setValue(i,"morphTargetsTexture",m.texture,t),u.getUniforms().setValue(i,"morphTargetsTextureSize",m.size)}return{update:s}}function bp(i,e,t,n){let r=new WeakMap;function s(u){const l=n.render.frame,h=u.geometry,p=e.get(u,h);if(r.get(p)!==l&&(e.update(p),r.set(p,l)),u.isInstancedMesh&&(u.hasEventListener("dispose",o)===!1&&u.addEventListener("dispose",o),r.get(u)!==l&&(t.update(u.instanceMatrix,i.ARRAY_BUFFER),u.instanceColor!==null&&t.update(u.instanceColor,i.ARRAY_BUFFER),r.set(u,l))),u.isSkinnedMesh){const m=u.skeleton;r.get(m)!==l&&(m.update(),r.set(m,l))}return p}function a(){r=new WeakMap}function o(u){const l=u.target;l.removeEventListener("dispose",o),t.remove(l.instanceMatrix),l.instanceColor!==null&&t.remove(l.instanceColor)}return{update:s,dispose:a}}const Tp={[yl]:"LINEAR_TONE_MAPPING",[El]:"REINHARD_TONE_MAPPING",[bl]:"CINEON_TONE_MAPPING",[Tl]:"ACES_FILMIC_TONE_MAPPING",[wl]:"AGX_TONE_MAPPING",[Rl]:"NEUTRAL_TONE_MAPPING",[Al]:"CUSTOM_TONE_MAPPING"};function Ap(i,e,t,n,r){const s=new Dn(e,t,{type:i,depthBuffer:n,stencilBuffer:r}),a=new Dn(e,t,{type:Yn,depthBuffer:!1,stencilBuffer:!1}),o=new bn;o.setAttribute("position",new gn([-1,3,0,-1,-1,0,3,-1,0],3)),o.setAttribute("uv",new gn([0,2,0,0,2,0],2));const u=new _f({uniforms:{tDiffuse:{value:null}},vertexShader:`
			precision highp float;

			uniform mat4 modelViewMatrix;
			uniform mat4 projectionMatrix;

			attribute vec3 position;
			attribute vec2 uv;

			varying vec2 vUv;

			void main() {
				vUv = uv;
				gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
			}`,fragmentShader:`
			precision highp float;

			uniform sampler2D tDiffuse;

			varying vec2 vUv;

			#include <tonemapping_pars_fragment>
			#include <colorspace_pars_fragment>

			void main() {
				gl_FragColor = texture2D( tDiffuse, vUv );

				#ifdef LINEAR_TONE_MAPPING
					gl_FragColor.rgb = LinearToneMapping( gl_FragColor.rgb );
				#elif defined( REINHARD_TONE_MAPPING )
					gl_FragColor.rgb = ReinhardToneMapping( gl_FragColor.rgb );
				#elif defined( CINEON_TONE_MAPPING )
					gl_FragColor.rgb = CineonToneMapping( gl_FragColor.rgb );
				#elif defined( ACES_FILMIC_TONE_MAPPING )
					gl_FragColor.rgb = ACESFilmicToneMapping( gl_FragColor.rgb );
				#elif defined( AGX_TONE_MAPPING )
					gl_FragColor.rgb = AgXToneMapping( gl_FragColor.rgb );
				#elif defined( NEUTRAL_TONE_MAPPING )
					gl_FragColor.rgb = NeutralToneMapping( gl_FragColor.rgb );
				#elif defined( CUSTOM_TONE_MAPPING )
					gl_FragColor.rgb = CustomToneMapping( gl_FragColor.rgb );
				#endif

				#ifdef SRGB_TRANSFER
					gl_FragColor = sRGBTransferOETF( gl_FragColor );
				#endif
			}`,depthTest:!1,depthWrite:!1}),l=new In(o,u),h=new Kl(-1,1,1,-1,0,1);let p=null,m=null,v=!1,y,b=null,g=[],d=!1;this.setSize=function(R,C){s.setSize(R,C),a.setSize(R,C);for(let A=0;A<g.length;A++){const P=g[A];P.setSize&&P.setSize(R,C)}},this.setEffects=function(R){g=R,d=g.length>0&&g[0].isRenderPass===!0;const C=s.width,A=s.height;for(let P=0;P<g.length;P++){const D=g[P];D.setSize&&D.setSize(C,A)}},this.begin=function(R,C){if(v||R.toneMapping===Pn&&g.length===0)return!1;if(b=C,C!==null){const A=C.width,P=C.height;(s.width!==A||s.height!==P)&&this.setSize(A,P)}return d===!1&&R.setRenderTarget(s),y=R.toneMapping,R.toneMapping=Pn,!0},this.hasRenderPass=function(){return d},this.end=function(R,C){R.toneMapping=y,v=!0;let A=s,P=a;for(let D=0;D<g.length;D++){const I=g[D];if(I.enabled!==!1&&(I.render(R,P,A,C),I.needsSwap!==!1)){const G=A;A=P,P=G}}if(p!==R.outputColorSpace||m!==R.toneMapping){p=R.outputColorSpace,m=R.toneMapping,u.defines={},_t.getTransfer(p)===wt&&(u.defines.SRGB_TRANSFER="");const D=Tp[m];D&&(u.defines[D]=""),u.needsUpdate=!0}u.uniforms.tDiffuse.value=A.texture,R.setRenderTarget(b),R.render(l,h),b=null,v=!1},this.isCompositing=function(){return v},this.dispose=function(){s.dispose(),a.dispose(),o.dispose(),u.dispose()}}const Zl=new rn,Wa=new mr(1,1),Jl=new Bl,Ql=new Wu,ec=new $l,il=[],rl=[],sl=new Float32Array(16),al=new Float32Array(9),ol=new Float32Array(4);function ji(i,e,t){const n=i[0];if(n<=0||n>0)return i;const r=e*t;let s=il[r];if(s===void 0&&(s=new Float32Array(r),il[r]=s),e!==0){n.toArray(s,0);for(let a=1,o=0;a!==e;++a)o+=t,i[a].toArray(s,o)}return s}function Gt(i,e){if(i.length!==e.length)return!1;for(let t=0,n=i.length;t<n;t++)if(i[t]!==e[t])return!1;return!0}function Ht(i,e){for(let t=0,n=e.length;t<n;t++)i[t]=e[t]}function ls(i,e){let t=rl[e];t===void 0&&(t=new Int32Array(e),rl[e]=t);for(let n=0;n!==e;++n)t[n]=i.allocateTextureUnit();return t}function wp(i,e){const t=this.cache;t[0]!==e&&(i.uniform1f(this.addr,e),t[0]=e)}function Rp(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2f(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Gt(t,e))return;i.uniform2fv(this.addr,e),Ht(t,e)}}function Cp(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3f(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else if(e.r!==void 0)(t[0]!==e.r||t[1]!==e.g||t[2]!==e.b)&&(i.uniform3f(this.addr,e.r,e.g,e.b),t[0]=e.r,t[1]=e.g,t[2]=e.b);else{if(Gt(t,e))return;i.uniform3fv(this.addr,e),Ht(t,e)}}function Pp(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4f(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Gt(t,e))return;i.uniform4fv(this.addr,e),Ht(t,e)}}function Dp(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(Gt(t,e))return;i.uniformMatrix2fv(this.addr,!1,e),Ht(t,e)}else{if(Gt(t,n))return;ol.set(n),i.uniformMatrix2fv(this.addr,!1,ol),Ht(t,n)}}function Lp(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(Gt(t,e))return;i.uniformMatrix3fv(this.addr,!1,e),Ht(t,e)}else{if(Gt(t,n))return;al.set(n),i.uniformMatrix3fv(this.addr,!1,al),Ht(t,n)}}function Up(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(Gt(t,e))return;i.uniformMatrix4fv(this.addr,!1,e),Ht(t,e)}else{if(Gt(t,n))return;sl.set(n),i.uniformMatrix4fv(this.addr,!1,sl),Ht(t,n)}}function Ip(i,e){const t=this.cache;t[0]!==e&&(i.uniform1i(this.addr,e),t[0]=e)}function Fp(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2i(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Gt(t,e))return;i.uniform2iv(this.addr,e),Ht(t,e)}}function Np(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3i(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(Gt(t,e))return;i.uniform3iv(this.addr,e),Ht(t,e)}}function Op(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4i(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Gt(t,e))return;i.uniform4iv(this.addr,e),Ht(t,e)}}function Bp(i,e){const t=this.cache;t[0]!==e&&(i.uniform1ui(this.addr,e),t[0]=e)}function Vp(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2ui(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Gt(t,e))return;i.uniform2uiv(this.addr,e),Ht(t,e)}}function zp(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3ui(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(Gt(t,e))return;i.uniform3uiv(this.addr,e),Ht(t,e)}}function Gp(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4ui(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Gt(t,e))return;i.uniform4uiv(this.addr,e),Ht(t,e)}}function Hp(i,e,t){const n=this.cache,r=t.allocateTextureUnit();n[0]!==r&&(i.uniform1i(this.addr,r),n[0]=r);let s;this.type===i.SAMPLER_2D_SHADOW?(Wa.compareFunction=t.isReversedDepthBuffer()?io:no,s=Wa):s=Zl,t.setTexture2D(e||s,r)}function kp(i,e,t){const n=this.cache,r=t.allocateTextureUnit();n[0]!==r&&(i.uniform1i(this.addr,r),n[0]=r),t.setTexture3D(e||Ql,r)}function Wp(i,e,t){const n=this.cache,r=t.allocateTextureUnit();n[0]!==r&&(i.uniform1i(this.addr,r),n[0]=r),t.setTextureCube(e||ec,r)}function Xp(i,e,t){const n=this.cache,r=t.allocateTextureUnit();n[0]!==r&&(i.uniform1i(this.addr,r),n[0]=r),t.setTexture2DArray(e||Jl,r)}function $p(i){switch(i){case 5126:return wp;case 35664:return Rp;case 35665:return Cp;case 35666:return Pp;case 35674:return Dp;case 35675:return Lp;case 35676:return Up;case 5124:case 35670:return Ip;case 35667:case 35671:return Fp;case 35668:case 35672:return Np;case 35669:case 35673:return Op;case 5125:return Bp;case 36294:return Vp;case 36295:return zp;case 36296:return Gp;case 35678:case 36198:case 36298:case 36306:case 35682:return Hp;case 35679:case 36299:case 36307:return kp;case 35680:case 36300:case 36308:case 36293:return Wp;case 36289:case 36303:case 36311:case 36292:return Xp}}function qp(i,e){i.uniform1fv(this.addr,e)}function Yp(i,e){const t=ji(e,this.size,2);i.uniform2fv(this.addr,t)}function Kp(i,e){const t=ji(e,this.size,3);i.uniform3fv(this.addr,t)}function jp(i,e){const t=ji(e,this.size,4);i.uniform4fv(this.addr,t)}function Zp(i,e){const t=ji(e,this.size,4);i.uniformMatrix2fv(this.addr,!1,t)}function Jp(i,e){const t=ji(e,this.size,9);i.uniformMatrix3fv(this.addr,!1,t)}function Qp(i,e){const t=ji(e,this.size,16);i.uniformMatrix4fv(this.addr,!1,t)}function em(i,e){i.uniform1iv(this.addr,e)}function tm(i,e){i.uniform2iv(this.addr,e)}function nm(i,e){i.uniform3iv(this.addr,e)}function im(i,e){i.uniform4iv(this.addr,e)}function rm(i,e){i.uniform1uiv(this.addr,e)}function sm(i,e){i.uniform2uiv(this.addr,e)}function am(i,e){i.uniform3uiv(this.addr,e)}function om(i,e){i.uniform4uiv(this.addr,e)}function lm(i,e,t){const n=this.cache,r=e.length,s=ls(t,r);Gt(n,s)||(i.uniform1iv(this.addr,s),Ht(n,s));let a;this.type===i.SAMPLER_2D_SHADOW?a=Wa:a=Zl;for(let o=0;o!==r;++o)t.setTexture2D(e[o]||a,s[o])}function cm(i,e,t){const n=this.cache,r=e.length,s=ls(t,r);Gt(n,s)||(i.uniform1iv(this.addr,s),Ht(n,s));for(let a=0;a!==r;++a)t.setTexture3D(e[a]||Ql,s[a])}function um(i,e,t){const n=this.cache,r=e.length,s=ls(t,r);Gt(n,s)||(i.uniform1iv(this.addr,s),Ht(n,s));for(let a=0;a!==r;++a)t.setTextureCube(e[a]||ec,s[a])}function fm(i,e,t){const n=this.cache,r=e.length,s=ls(t,r);Gt(n,s)||(i.uniform1iv(this.addr,s),Ht(n,s));for(let a=0;a!==r;++a)t.setTexture2DArray(e[a]||Jl,s[a])}function hm(i){switch(i){case 5126:return qp;case 35664:return Yp;case 35665:return Kp;case 35666:return jp;case 35674:return Zp;case 35675:return Jp;case 35676:return Qp;case 5124:case 35670:return em;case 35667:case 35671:return tm;case 35668:case 35672:return nm;case 35669:case 35673:return im;case 5125:return rm;case 36294:return sm;case 36295:return am;case 36296:return om;case 35678:case 36198:case 36298:case 36306:case 35682:return lm;case 35679:case 36299:case 36307:return cm;case 35680:case 36300:case 36308:case 36293:return um;case 36289:case 36303:case 36311:case 36292:return fm}}class dm{constructor(e,t,n){this.id=e,this.addr=n,this.cache=[],this.type=t.type,this.setValue=$p(t.type)}}class pm{constructor(e,t,n){this.id=e,this.addr=n,this.cache=[],this.type=t.type,this.size=t.size,this.setValue=hm(t.type)}}class mm{constructor(e){this.id=e,this.seq=[],this.map={}}setValue(e,t,n){const r=this.seq;for(let s=0,a=r.length;s!==a;++s){const o=r[s];o.setValue(e,t[o.id],n)}}}const js=/(\w+)(\])?(\[|\.)?/g;function ll(i,e){i.seq.push(e),i.map[e.id]=e}function _m(i,e,t){const n=i.name,r=n.length;for(js.lastIndex=0;;){const s=js.exec(n),a=js.lastIndex;let o=s[1];const u=s[2]==="]",l=s[3];if(u&&(o=o|0),l===void 0||l==="["&&a+2===r){ll(t,l===void 0?new dm(o,i,e):new pm(o,i,e));break}else{let p=t.map[o];p===void 0&&(p=new mm(o),ll(t,p)),t=p}}}class ts{constructor(e,t){this.seq=[],this.map={};const n=e.getProgramParameter(t,e.ACTIVE_UNIFORMS);for(let a=0;a<n;++a){const o=e.getActiveUniform(t,a),u=e.getUniformLocation(t,o.name);_m(o,u,this)}const r=[],s=[];for(const a of this.seq)a.type===e.SAMPLER_2D_SHADOW||a.type===e.SAMPLER_CUBE_SHADOW||a.type===e.SAMPLER_2D_ARRAY_SHADOW?r.push(a):s.push(a);r.length>0&&(this.seq=r.concat(s))}setValue(e,t,n,r){const s=this.map[t];s!==void 0&&s.setValue(e,n,r)}setOptional(e,t,n){const r=t[n];r!==void 0&&this.setValue(e,n,r)}static upload(e,t,n,r){for(let s=0,a=t.length;s!==a;++s){const o=t[s],u=n[o.id];u.needsUpdate!==!1&&o.setValue(e,u.value,r)}}static seqWithValue(e,t){const n=[];for(let r=0,s=e.length;r!==s;++r){const a=e[r];a.id in t&&n.push(a)}return n}}function cl(i,e,t){const n=i.createShader(e);return i.shaderSource(n,t),i.compileShader(n),n}const gm=37297;let vm=0;function xm(i,e){const t=i.split(`
`),n=[],r=Math.max(e-6,0),s=Math.min(e+6,t.length);for(let a=r;a<s;a++){const o=a+1;n.push(`${o===e?">":" "} ${o}: ${t[a]}`)}return n.join(`
`)}const ul=new st;function Mm(i){_t._getMatrix(ul,_t.workingColorSpace,i);const e=`mat3( ${ul.elements.map(t=>t.toFixed(4))} )`;switch(_t.getTransfer(i)){case ns:return[e,"LinearTransferOETF"];case wt:return[e,"sRGBTransferOETF"];default:return et("WebGLProgram: Unsupported color space: ",i),[e,"LinearTransferOETF"]}}function fl(i,e,t){const n=i.getShaderParameter(e,i.COMPILE_STATUS),s=(i.getShaderInfoLog(e)||"").trim();if(n&&s==="")return"";const a=/ERROR: 0:(\d+)/.exec(s);if(a){const o=parseInt(a[1]);return t.toUpperCase()+`

`+s+`

`+xm(i.getShaderSource(e),o)}else return s}function Sm(i,e){const t=Mm(e);return[`vec4 ${i}( vec4 value ) {`,`	return ${t[1]}( vec4( value.rgb * ${t[0]}, value.a ) );`,"}"].join(`
`)}const ym={[yl]:"Linear",[El]:"Reinhard",[bl]:"Cineon",[Tl]:"ACESFilmic",[wl]:"AgX",[Rl]:"Neutral",[Al]:"Custom"};function Em(i,e){const t=ym[e];return t===void 0?(et("WebGLProgram: Unsupported toneMapping:",e),"vec3 "+i+"( vec3 color ) { return LinearToneMapping( color ); }"):"vec3 "+i+"( vec3 color ) { return "+t+"ToneMapping( color ); }"}const Kr=new $;function bm(){_t.getLuminanceCoefficients(Kr);const i=Kr.x.toFixed(4),e=Kr.y.toFixed(4),t=Kr.z.toFixed(4);return["float luminance( const in vec3 rgb ) {",`	const vec3 weights = vec3( ${i}, ${e}, ${t} );`,"	return dot( weights, rgb );","}"].join(`
`)}function Tm(i){return[i.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":"",i.extensionMultiDraw?"#extension GL_ANGLE_multi_draw : require":""].filter(fr).join(`
`)}function Am(i){const e=[];for(const t in i){const n=i[t];n!==!1&&e.push("#define "+t+" "+n)}return e.join(`
`)}function wm(i,e){const t={},n=i.getProgramParameter(e,i.ACTIVE_ATTRIBUTES);for(let r=0;r<n;r++){const s=i.getActiveAttrib(e,r),a=s.name;let o=1;s.type===i.FLOAT_MAT2&&(o=2),s.type===i.FLOAT_MAT3&&(o=3),s.type===i.FLOAT_MAT4&&(o=4),t[a]={type:s.type,location:i.getAttribLocation(e,a),locationSize:o}}return t}function fr(i){return i!==""}function hl(i,e){const t=e.numSpotLightShadows+e.numSpotLightMaps-e.numSpotLightShadowsWithMaps;return i.replace(/NUM_DIR_LIGHTS/g,e.numDirLights).replace(/NUM_SPOT_LIGHTS/g,e.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,e.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,t).replace(/NUM_RECT_AREA_LIGHTS/g,e.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,e.numPointLights).replace(/NUM_HEMI_LIGHTS/g,e.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,e.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,e.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,e.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,e.numPointLightShadows)}function dl(i,e){return i.replace(/NUM_CLIPPING_PLANES/g,e.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,e.numClippingPlanes-e.numClipIntersection)}const Rm=/^[ \t]*#include +<([\w\d./]+)>/gm;function Xa(i){return i.replace(Rm,Pm)}const Cm=new Map;function Pm(i,e){let t=ct[e];if(t===void 0){const n=Cm.get(e);if(n!==void 0)t=ct[n],et('WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.',e,n);else throw new Error("Can not resolve #include <"+e+">")}return Xa(t)}const Dm=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function pl(i){return i.replace(Dm,Lm)}function Lm(i,e,t,n){let r="";for(let s=parseInt(e);s<parseInt(t);s++)r+=n.replace(/\[\s*i\s*\]/g,"[ "+s+" ]").replace(/UNROLLED_LOOP_INDEX/g,s);return r}function ml(i){let e=`precision ${i.precision} float;
	precision ${i.precision} int;
	precision ${i.precision} sampler2D;
	precision ${i.precision} samplerCube;
	precision ${i.precision} sampler3D;
	precision ${i.precision} sampler2DArray;
	precision ${i.precision} sampler2DShadow;
	precision ${i.precision} samplerCubeShadow;
	precision ${i.precision} sampler2DArrayShadow;
	precision ${i.precision} isampler2D;
	precision ${i.precision} isampler3D;
	precision ${i.precision} isamplerCube;
	precision ${i.precision} isampler2DArray;
	precision ${i.precision} usampler2D;
	precision ${i.precision} usampler3D;
	precision ${i.precision} usamplerCube;
	precision ${i.precision} usampler2DArray;
	`;return i.precision==="highp"?e+=`
#define HIGH_PRECISION`:i.precision==="mediump"?e+=`
#define MEDIUM_PRECISION`:i.precision==="lowp"&&(e+=`
#define LOW_PRECISION`),e}const Um={[jr]:"SHADOWMAP_TYPE_PCF",[ur]:"SHADOWMAP_TYPE_VSM"};function Im(i){return Um[i.shadowMapType]||"SHADOWMAP_TYPE_BASIC"}const Fm={[Si]:"ENVMAP_TYPE_CUBE",[ki]:"ENVMAP_TYPE_CUBE",[ss]:"ENVMAP_TYPE_CUBE_UV"};function Nm(i){return i.envMap===!1?"ENVMAP_TYPE_CUBE":Fm[i.envMapMode]||"ENVMAP_TYPE_CUBE"}const Om={[ki]:"ENVMAP_MODE_REFRACTION"};function Bm(i){return i.envMap===!1?"ENVMAP_MODE_REFLECTION":Om[i.envMapMode]||"ENVMAP_MODE_REFLECTION"}const Vm={[Ya]:"ENVMAP_BLENDING_MULTIPLY",[Tu]:"ENVMAP_BLENDING_MIX",[Au]:"ENVMAP_BLENDING_ADD"};function zm(i){return i.envMap===!1?"ENVMAP_BLENDING_NONE":Vm[i.combine]||"ENVMAP_BLENDING_NONE"}function Gm(i){const e=i.envMapCubeUVHeight;if(e===null)return null;const t=Math.log2(e)-2,n=1/e;return{texelWidth:1/(3*Math.max(Math.pow(2,t),112)),texelHeight:n,maxMip:t}}function Hm(i,e,t,n){const r=i.getContext(),s=t.defines;let a=t.vertexShader,o=t.fragmentShader;const u=Im(t),l=Nm(t),h=Bm(t),p=zm(t),m=Gm(t),v=Tm(t),y=Am(s),b=r.createProgram();let g,d,R=t.glslVersion?"#version "+t.glslVersion+`
`:"";t.isRawShaderMaterial?(g=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,y].filter(fr).join(`
`),g.length>0&&(g+=`
`),d=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,y].filter(fr).join(`
`),d.length>0&&(d+=`
`)):(g=[ml(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,y,t.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",t.batching?"#define USE_BATCHING":"",t.batchingColor?"#define USE_BATCHING_COLOR":"",t.instancing?"#define USE_INSTANCING":"",t.instancingColor?"#define USE_INSTANCING_COLOR":"",t.instancingMorph?"#define USE_INSTANCING_MORPH":"",t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.map?"#define USE_MAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+h:"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.displacementMap?"#define USE_DISPLACEMENTMAP":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.mapUv?"#define MAP_UV "+t.mapUv:"",t.alphaMapUv?"#define ALPHAMAP_UV "+t.alphaMapUv:"",t.lightMapUv?"#define LIGHTMAP_UV "+t.lightMapUv:"",t.aoMapUv?"#define AOMAP_UV "+t.aoMapUv:"",t.emissiveMapUv?"#define EMISSIVEMAP_UV "+t.emissiveMapUv:"",t.bumpMapUv?"#define BUMPMAP_UV "+t.bumpMapUv:"",t.normalMapUv?"#define NORMALMAP_UV "+t.normalMapUv:"",t.displacementMapUv?"#define DISPLACEMENTMAP_UV "+t.displacementMapUv:"",t.metalnessMapUv?"#define METALNESSMAP_UV "+t.metalnessMapUv:"",t.roughnessMapUv?"#define ROUGHNESSMAP_UV "+t.roughnessMapUv:"",t.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+t.anisotropyMapUv:"",t.clearcoatMapUv?"#define CLEARCOATMAP_UV "+t.clearcoatMapUv:"",t.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+t.clearcoatNormalMapUv:"",t.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+t.clearcoatRoughnessMapUv:"",t.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+t.iridescenceMapUv:"",t.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+t.iridescenceThicknessMapUv:"",t.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+t.sheenColorMapUv:"",t.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+t.sheenRoughnessMapUv:"",t.specularMapUv?"#define SPECULARMAP_UV "+t.specularMapUv:"",t.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+t.specularColorMapUv:"",t.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+t.specularIntensityMapUv:"",t.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+t.transmissionMapUv:"",t.thicknessMapUv?"#define THICKNESSMAP_UV "+t.thicknessMapUv:"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors?"#define USE_COLOR":"",t.vertexAlphas?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.flatShading?"#define FLAT_SHADED":"",t.skinning?"#define USE_SKINNING":"",t.morphTargets?"#define USE_MORPHTARGETS":"",t.morphNormals&&t.flatShading===!1?"#define USE_MORPHNORMALS":"",t.morphColors?"#define USE_MORPHCOLORS":"",t.morphTargetsCount>0?"#define MORPHTARGETS_TEXTURE_STRIDE "+t.morphTextureStride:"",t.morphTargetsCount>0?"#define MORPHTARGETS_COUNT "+t.morphTargetsCount:"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+u:"",t.sizeAttenuation?"#define USE_SIZEATTENUATION":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","#ifdef USE_INSTANCING_MORPH","	uniform sampler2D morphTexture;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(fr).join(`
`),d=[ml(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,y,t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.alphaToCoverage?"#define ALPHA_TO_COVERAGE":"",t.map?"#define USE_MAP":"",t.matcap?"#define USE_MATCAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+l:"",t.envMap?"#define "+h:"",t.envMap?"#define "+p:"",m?"#define CUBEUV_TEXEL_WIDTH "+m.texelWidth:"",m?"#define CUBEUV_TEXEL_HEIGHT "+m.texelHeight:"",m?"#define CUBEUV_MAX_MIP "+m.maxMip+".0":"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoat?"#define USE_CLEARCOAT":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.dispersion?"#define USE_DISPERSION":"",t.iridescence?"#define USE_IRIDESCENCE":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaTest?"#define USE_ALPHATEST":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.sheen?"#define USE_SHEEN":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors||t.instancingColor||t.batchingColor?"#define USE_COLOR":"",t.vertexAlphas?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.gradientMap?"#define USE_GRADIENTMAP":"",t.flatShading?"#define FLAT_SHADED":"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+u:"",t.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",t.decodeVideoTextureEmissive?"#define DECODE_VIDEO_TEXTURE_EMISSIVE":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",t.toneMapping!==Pn?"#define TONE_MAPPING":"",t.toneMapping!==Pn?ct.tonemapping_pars_fragment:"",t.toneMapping!==Pn?Em("toneMapping",t.toneMapping):"",t.dithering?"#define DITHERING":"",t.opaque?"#define OPAQUE":"",ct.colorspace_pars_fragment,Sm("linearToOutputTexel",t.outputColorSpace),bm(),t.useDepthPacking?"#define DEPTH_PACKING "+t.depthPacking:"",`
`].filter(fr).join(`
`)),a=Xa(a),a=hl(a,t),a=dl(a,t),o=Xa(o),o=hl(o,t),o=dl(o,t),a=pl(a),o=pl(o),t.isRawShaderMaterial!==!0&&(R=`#version 300 es
`,g=[v,"#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+g,d=["#define varying in",t.glslVersion===Ro?"":"layout(location = 0) out highp vec4 pc_fragColor;",t.glslVersion===Ro?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+d);const C=R+g+a,A=R+d+o,P=cl(r,r.VERTEX_SHADER,C),D=cl(r,r.FRAGMENT_SHADER,A);r.attachShader(b,P),r.attachShader(b,D),t.index0AttributeName!==void 0?r.bindAttribLocation(b,0,t.index0AttributeName):t.morphTargets===!0&&r.bindAttribLocation(b,0,"position"),r.linkProgram(b);function I(B){if(i.debug.checkShaderErrors){const j=r.getProgramInfoLog(b)||"",K=r.getShaderInfoLog(P)||"",ie=r.getShaderInfoLog(D)||"",oe=j.trim(),Q=K.trim(),te=ie.trim();let ue=!0,Ae=!0;if(r.getProgramParameter(b,r.LINK_STATUS)===!1)if(ue=!1,typeof i.debug.onShaderError=="function")i.debug.onShaderError(r,b,P,D);else{const Ee=fl(r,P,"vertex"),Le=fl(r,D,"fragment");xt("THREE.WebGLProgram: Shader Error "+r.getError()+" - VALIDATE_STATUS "+r.getProgramParameter(b,r.VALIDATE_STATUS)+`

Material Name: `+B.name+`
Material Type: `+B.type+`

Program Info Log: `+oe+`
`+Ee+`
`+Le)}else oe!==""?et("WebGLProgram: Program Info Log:",oe):(Q===""||te==="")&&(Ae=!1);Ae&&(B.diagnostics={runnable:ue,programLog:oe,vertexShader:{log:Q,prefix:g},fragmentShader:{log:te,prefix:d}})}r.deleteShader(P),r.deleteShader(D),G=new ts(r,b),x=wm(r,b)}let G;this.getUniforms=function(){return G===void 0&&I(this),G};let x;this.getAttributes=function(){return x===void 0&&I(this),x};let E=t.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return E===!1&&(E=r.getProgramParameter(b,gm)),E},this.destroy=function(){n.releaseStatesOfProgram(this),r.deleteProgram(b),this.program=void 0},this.type=t.shaderType,this.name=t.shaderName,this.id=vm++,this.cacheKey=e,this.usedTimes=1,this.program=b,this.vertexShader=P,this.fragmentShader=D,this}let km=0;class Wm{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(e){const t=e.vertexShader,n=e.fragmentShader,r=this._getShaderStage(t),s=this._getShaderStage(n),a=this._getShaderCacheForMaterial(e);return a.has(r)===!1&&(a.add(r),r.usedTimes++),a.has(s)===!1&&(a.add(s),s.usedTimes++),this}remove(e){const t=this.materialCache.get(e);for(const n of t)n.usedTimes--,n.usedTimes===0&&this.shaderCache.delete(n.code);return this.materialCache.delete(e),this}getVertexShaderID(e){return this._getShaderStage(e.vertexShader).id}getFragmentShaderID(e){return this._getShaderStage(e.fragmentShader).id}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(e){const t=this.materialCache;let n=t.get(e);return n===void 0&&(n=new Set,t.set(e,n)),n}_getShaderStage(e){const t=this.shaderCache;let n=t.get(e);return n===void 0&&(n=new Xm(e),t.set(e,n)),n}}class Xm{constructor(e){this.id=km++,this.code=e,this.usedTimes=0}}function $m(i,e,t,n,r,s,a){const o=new Vl,u=new Wm,l=new Set,h=[],p=new Map,m=r.logarithmicDepthBuffer;let v=r.precision;const y={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distance",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function b(x){return l.add(x),x===0?"uv":`uv${x}`}function g(x,E,B,j,K){const ie=j.fog,oe=K.geometry,Q=x.isMeshStandardMaterial?j.environment:null,te=(x.isMeshStandardMaterial?t:e).get(x.envMap||Q),ue=te&&te.mapping===ss?te.image.height:null,Ae=y[x.type];x.precision!==null&&(v=r.getMaxPrecision(x.precision),v!==x.precision&&et("WebGLProgram.getParameters:",x.precision,"not supported, using",v,"instead."));const Ee=oe.morphAttributes.position||oe.morphAttributes.normal||oe.morphAttributes.color,Le=Ee!==void 0?Ee.length:0;let at=0;oe.morphAttributes.position!==void 0&&(at=1),oe.morphAttributes.normal!==void 0&&(at=2),oe.morphAttributes.color!==void 0&&(at=3);let it,Ut,Dt,re;if(Ae){const vt=wn[Ae];it=vt.vertexShader,Ut=vt.fragmentShader}else it=x.vertexShader,Ut=x.fragmentShader,u.update(x),Dt=u.getVertexShaderID(x),re=u.getFragmentShaderID(x);const fe=i.getRenderTarget(),Ne=i.state.buffers.depth.getReversed(),Qe=K.isInstancedMesh===!0,Ve=K.isBatchedMesh===!0,ht=!!x.map,Vt=!!x.matcap,Je=!!te,ut=!!x.aoMap,Tt=!!x.lightMap,nt=!!x.bumpMap,Ft=!!x.normalMap,F=!!x.displacementMap,It=!!x.emissiveMap,mt=!!x.metalnessMap,gt=!!x.roughnessMap,Oe=x.anisotropy>0,w=x.clearcoat>0,_=x.dispersion>0,V=x.iridescence>0,ne=x.sheen>0,se=x.transmission>0,ee=Oe&&!!x.anisotropyMap,Me=w&&!!x.clearcoatMap,ve=w&&!!x.clearcoatNormalMap,me=w&&!!x.clearcoatRoughnessMap,Ye=V&&!!x.iridescenceMap,ce=V&&!!x.iridescenceThicknessMap,Se=ne&&!!x.sheenColorMap,Be=ne&&!!x.sheenRoughnessMap,Ge=!!x.specularMap,ye=!!x.specularColorMap,tt=!!x.specularIntensityMap,O=se&&!!x.transmissionMap,we=se&&!!x.thicknessMap,_e=!!x.gradientMap,De=!!x.alphaMap,de=x.alphaTest>0,le=!!x.alphaHash,xe=!!x.extensions;let Ke=Pn;x.toneMapped&&(fe===null||fe.isXRRenderTarget===!0)&&(Ke=i.toneMapping);const At={shaderID:Ae,shaderType:x.type,shaderName:x.name,vertexShader:it,fragmentShader:Ut,defines:x.defines,customVertexShaderID:Dt,customFragmentShaderID:re,isRawShaderMaterial:x.isRawShaderMaterial===!0,glslVersion:x.glslVersion,precision:v,batching:Ve,batchingColor:Ve&&K._colorsTexture!==null,instancing:Qe,instancingColor:Qe&&K.instanceColor!==null,instancingMorph:Qe&&K.morphTexture!==null,outputColorSpace:fe===null?i.outputColorSpace:fe.isXRRenderTarget===!0?fe.texture.colorSpace:Xi,alphaToCoverage:!!x.alphaToCoverage,map:ht,matcap:Vt,envMap:Je,envMapMode:Je&&te.mapping,envMapCubeUVHeight:ue,aoMap:ut,lightMap:Tt,bumpMap:nt,normalMap:Ft,displacementMap:F,emissiveMap:It,normalMapObjectSpace:Ft&&x.normalMapType===Cu,normalMapTangentSpace:Ft&&x.normalMapType===to,metalnessMap:mt,roughnessMap:gt,anisotropy:Oe,anisotropyMap:ee,clearcoat:w,clearcoatMap:Me,clearcoatNormalMap:ve,clearcoatRoughnessMap:me,dispersion:_,iridescence:V,iridescenceMap:Ye,iridescenceThicknessMap:ce,sheen:ne,sheenColorMap:Se,sheenRoughnessMap:Be,specularMap:Ge,specularColorMap:ye,specularIntensityMap:tt,transmission:se,transmissionMap:O,thicknessMap:we,gradientMap:_e,opaque:x.transparent===!1&&x.blending===zi&&x.alphaToCoverage===!1,alphaMap:De,alphaTest:de,alphaHash:le,combine:x.combine,mapUv:ht&&b(x.map.channel),aoMapUv:ut&&b(x.aoMap.channel),lightMapUv:Tt&&b(x.lightMap.channel),bumpMapUv:nt&&b(x.bumpMap.channel),normalMapUv:Ft&&b(x.normalMap.channel),displacementMapUv:F&&b(x.displacementMap.channel),emissiveMapUv:It&&b(x.emissiveMap.channel),metalnessMapUv:mt&&b(x.metalnessMap.channel),roughnessMapUv:gt&&b(x.roughnessMap.channel),anisotropyMapUv:ee&&b(x.anisotropyMap.channel),clearcoatMapUv:Me&&b(x.clearcoatMap.channel),clearcoatNormalMapUv:ve&&b(x.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:me&&b(x.clearcoatRoughnessMap.channel),iridescenceMapUv:Ye&&b(x.iridescenceMap.channel),iridescenceThicknessMapUv:ce&&b(x.iridescenceThicknessMap.channel),sheenColorMapUv:Se&&b(x.sheenColorMap.channel),sheenRoughnessMapUv:Be&&b(x.sheenRoughnessMap.channel),specularMapUv:Ge&&b(x.specularMap.channel),specularColorMapUv:ye&&b(x.specularColorMap.channel),specularIntensityMapUv:tt&&b(x.specularIntensityMap.channel),transmissionMapUv:O&&b(x.transmissionMap.channel),thicknessMapUv:we&&b(x.thicknessMap.channel),alphaMapUv:De&&b(x.alphaMap.channel),vertexTangents:!!oe.attributes.tangent&&(Ft||Oe),vertexColors:x.vertexColors,vertexAlphas:x.vertexColors===!0&&!!oe.attributes.color&&oe.attributes.color.itemSize===4,pointsUvs:K.isPoints===!0&&!!oe.attributes.uv&&(ht||De),fog:!!ie,useFog:x.fog===!0,fogExp2:!!ie&&ie.isFogExp2,flatShading:x.flatShading===!0&&x.wireframe===!1,sizeAttenuation:x.sizeAttenuation===!0,logarithmicDepthBuffer:m,reversedDepthBuffer:Ne,skinning:K.isSkinnedMesh===!0,morphTargets:oe.morphAttributes.position!==void 0,morphNormals:oe.morphAttributes.normal!==void 0,morphColors:oe.morphAttributes.color!==void 0,morphTargetsCount:Le,morphTextureStride:at,numDirLights:E.directional.length,numPointLights:E.point.length,numSpotLights:E.spot.length,numSpotLightMaps:E.spotLightMap.length,numRectAreaLights:E.rectArea.length,numHemiLights:E.hemi.length,numDirLightShadows:E.directionalShadowMap.length,numPointLightShadows:E.pointShadowMap.length,numSpotLightShadows:E.spotShadowMap.length,numSpotLightShadowsWithMaps:E.numSpotLightShadowsWithMaps,numLightProbes:E.numLightProbes,numClippingPlanes:a.numPlanes,numClipIntersection:a.numIntersection,dithering:x.dithering,shadowMapEnabled:i.shadowMap.enabled&&B.length>0,shadowMapType:i.shadowMap.type,toneMapping:Ke,decodeVideoTexture:ht&&x.map.isVideoTexture===!0&&_t.getTransfer(x.map.colorSpace)===wt,decodeVideoTextureEmissive:It&&x.emissiveMap.isVideoTexture===!0&&_t.getTransfer(x.emissiveMap.colorSpace)===wt,premultipliedAlpha:x.premultipliedAlpha,doubleSided:x.side===Wn,flipSided:x.side===on,useDepthPacking:x.depthPacking>=0,depthPacking:x.depthPacking||0,index0AttributeName:x.index0AttributeName,extensionClipCullDistance:xe&&x.extensions.clipCullDistance===!0&&n.has("WEBGL_clip_cull_distance"),extensionMultiDraw:(xe&&x.extensions.multiDraw===!0||Ve)&&n.has("WEBGL_multi_draw"),rendererExtensionParallelShaderCompile:n.has("KHR_parallel_shader_compile"),customProgramCacheKey:x.customProgramCacheKey()};return At.vertexUv1s=l.has(1),At.vertexUv2s=l.has(2),At.vertexUv3s=l.has(3),l.clear(),At}function d(x){const E=[];if(x.shaderID?E.push(x.shaderID):(E.push(x.customVertexShaderID),E.push(x.customFragmentShaderID)),x.defines!==void 0)for(const B in x.defines)E.push(B),E.push(x.defines[B]);return x.isRawShaderMaterial===!1&&(R(E,x),C(E,x),E.push(i.outputColorSpace)),E.push(x.customProgramCacheKey),E.join()}function R(x,E){x.push(E.precision),x.push(E.outputColorSpace),x.push(E.envMapMode),x.push(E.envMapCubeUVHeight),x.push(E.mapUv),x.push(E.alphaMapUv),x.push(E.lightMapUv),x.push(E.aoMapUv),x.push(E.bumpMapUv),x.push(E.normalMapUv),x.push(E.displacementMapUv),x.push(E.emissiveMapUv),x.push(E.metalnessMapUv),x.push(E.roughnessMapUv),x.push(E.anisotropyMapUv),x.push(E.clearcoatMapUv),x.push(E.clearcoatNormalMapUv),x.push(E.clearcoatRoughnessMapUv),x.push(E.iridescenceMapUv),x.push(E.iridescenceThicknessMapUv),x.push(E.sheenColorMapUv),x.push(E.sheenRoughnessMapUv),x.push(E.specularMapUv),x.push(E.specularColorMapUv),x.push(E.specularIntensityMapUv),x.push(E.transmissionMapUv),x.push(E.thicknessMapUv),x.push(E.combine),x.push(E.fogExp2),x.push(E.sizeAttenuation),x.push(E.morphTargetsCount),x.push(E.morphAttributeCount),x.push(E.numDirLights),x.push(E.numPointLights),x.push(E.numSpotLights),x.push(E.numSpotLightMaps),x.push(E.numHemiLights),x.push(E.numRectAreaLights),x.push(E.numDirLightShadows),x.push(E.numPointLightShadows),x.push(E.numSpotLightShadows),x.push(E.numSpotLightShadowsWithMaps),x.push(E.numLightProbes),x.push(E.shadowMapType),x.push(E.toneMapping),x.push(E.numClippingPlanes),x.push(E.numClipIntersection),x.push(E.depthPacking)}function C(x,E){o.disableAll(),E.instancing&&o.enable(0),E.instancingColor&&o.enable(1),E.instancingMorph&&o.enable(2),E.matcap&&o.enable(3),E.envMap&&o.enable(4),E.normalMapObjectSpace&&o.enable(5),E.normalMapTangentSpace&&o.enable(6),E.clearcoat&&o.enable(7),E.iridescence&&o.enable(8),E.alphaTest&&o.enable(9),E.vertexColors&&o.enable(10),E.vertexAlphas&&o.enable(11),E.vertexUv1s&&o.enable(12),E.vertexUv2s&&o.enable(13),E.vertexUv3s&&o.enable(14),E.vertexTangents&&o.enable(15),E.anisotropy&&o.enable(16),E.alphaHash&&o.enable(17),E.batching&&o.enable(18),E.dispersion&&o.enable(19),E.batchingColor&&o.enable(20),E.gradientMap&&o.enable(21),x.push(o.mask),o.disableAll(),E.fog&&o.enable(0),E.useFog&&o.enable(1),E.flatShading&&o.enable(2),E.logarithmicDepthBuffer&&o.enable(3),E.reversedDepthBuffer&&o.enable(4),E.skinning&&o.enable(5),E.morphTargets&&o.enable(6),E.morphNormals&&o.enable(7),E.morphColors&&o.enable(8),E.premultipliedAlpha&&o.enable(9),E.shadowMapEnabled&&o.enable(10),E.doubleSided&&o.enable(11),E.flipSided&&o.enable(12),E.useDepthPacking&&o.enable(13),E.dithering&&o.enable(14),E.transmission&&o.enable(15),E.sheen&&o.enable(16),E.opaque&&o.enable(17),E.pointsUvs&&o.enable(18),E.decodeVideoTexture&&o.enable(19),E.decodeVideoTextureEmissive&&o.enable(20),E.alphaToCoverage&&o.enable(21),x.push(o.mask)}function A(x){const E=y[x.type];let B;if(E){const j=wn[E];B=sf.clone(j.uniforms)}else B=x.uniforms;return B}function P(x,E){let B=p.get(E);return B!==void 0?++B.usedTimes:(B=new Hm(i,E,x,s),h.push(B),p.set(E,B)),B}function D(x){if(--x.usedTimes===0){const E=h.indexOf(x);h[E]=h[h.length-1],h.pop(),p.delete(x.cacheKey),x.destroy()}}function I(x){u.remove(x)}function G(){u.dispose()}return{getParameters:g,getProgramCacheKey:d,getUniforms:A,acquireProgram:P,releaseProgram:D,releaseShaderCache:I,programs:h,dispose:G}}function qm(){let i=new WeakMap;function e(a){return i.has(a)}function t(a){let o=i.get(a);return o===void 0&&(o={},i.set(a,o)),o}function n(a){i.delete(a)}function r(a,o,u){i.get(a)[o]=u}function s(){i=new WeakMap}return{has:e,get:t,remove:n,update:r,dispose:s}}function Ym(i,e){return i.groupOrder!==e.groupOrder?i.groupOrder-e.groupOrder:i.renderOrder!==e.renderOrder?i.renderOrder-e.renderOrder:i.material.id!==e.material.id?i.material.id-e.material.id:i.z!==e.z?i.z-e.z:i.id-e.id}function _l(i,e){return i.groupOrder!==e.groupOrder?i.groupOrder-e.groupOrder:i.renderOrder!==e.renderOrder?i.renderOrder-e.renderOrder:i.z!==e.z?e.z-i.z:i.id-e.id}function gl(){const i=[];let e=0;const t=[],n=[],r=[];function s(){e=0,t.length=0,n.length=0,r.length=0}function a(p,m,v,y,b,g){let d=i[e];return d===void 0?(d={id:p.id,object:p,geometry:m,material:v,groupOrder:y,renderOrder:p.renderOrder,z:b,group:g},i[e]=d):(d.id=p.id,d.object=p,d.geometry=m,d.material=v,d.groupOrder=y,d.renderOrder=p.renderOrder,d.z=b,d.group=g),e++,d}function o(p,m,v,y,b,g){const d=a(p,m,v,y,b,g);v.transmission>0?n.push(d):v.transparent===!0?r.push(d):t.push(d)}function u(p,m,v,y,b,g){const d=a(p,m,v,y,b,g);v.transmission>0?n.unshift(d):v.transparent===!0?r.unshift(d):t.unshift(d)}function l(p,m){t.length>1&&t.sort(p||Ym),n.length>1&&n.sort(m||_l),r.length>1&&r.sort(m||_l)}function h(){for(let p=e,m=i.length;p<m;p++){const v=i[p];if(v.id===null)break;v.id=null,v.object=null,v.geometry=null,v.material=null,v.group=null}}return{opaque:t,transmissive:n,transparent:r,init:s,push:o,unshift:u,finish:h,sort:l}}function Km(){let i=new WeakMap;function e(n,r){const s=i.get(n);let a;return s===void 0?(a=new gl,i.set(n,[a])):r>=s.length?(a=new gl,s.push(a)):a=s[r],a}function t(){i=new WeakMap}return{get:e,dispose:t}}function jm(){const i={};return{get:function(e){if(i[e.id]!==void 0)return i[e.id];let t;switch(e.type){case"DirectionalLight":t={direction:new $,color:new yt};break;case"SpotLight":t={position:new $,direction:new $,color:new yt,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":t={position:new $,color:new yt,distance:0,decay:0};break;case"HemisphereLight":t={direction:new $,skyColor:new yt,groundColor:new yt};break;case"RectAreaLight":t={color:new yt,position:new $,halfWidth:new $,halfHeight:new $};break}return i[e.id]=t,t}}}function Zm(){const i={};return{get:function(e){if(i[e.id]!==void 0)return i[e.id];let t;switch(e.type){case"DirectionalLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new pt};break;case"SpotLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new pt};break;case"PointLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new pt,shadowCameraNear:1,shadowCameraFar:1e3};break}return i[e.id]=t,t}}}let Jm=0;function Qm(i,e){return(e.castShadow?2:0)-(i.castShadow?2:0)+(e.map?1:0)-(i.map?1:0)}function e_(i){const e=new jm,t=Zm(),n={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let l=0;l<9;l++)n.probe.push(new $);const r=new $,s=new Ot,a=new Ot;function o(l){let h=0,p=0,m=0;for(let x=0;x<9;x++)n.probe[x].set(0,0,0);let v=0,y=0,b=0,g=0,d=0,R=0,C=0,A=0,P=0,D=0,I=0;l.sort(Qm);for(let x=0,E=l.length;x<E;x++){const B=l[x],j=B.color,K=B.intensity,ie=B.distance;let oe=null;if(B.shadow&&B.shadow.map&&(B.shadow.map.texture.format===Wi?oe=B.shadow.map.texture:oe=B.shadow.map.depthTexture||B.shadow.map.texture),B.isAmbientLight)h+=j.r*K,p+=j.g*K,m+=j.b*K;else if(B.isLightProbe){for(let Q=0;Q<9;Q++)n.probe[Q].addScaledVector(B.sh.coefficients[Q],K);I++}else if(B.isDirectionalLight){const Q=e.get(B);if(Q.color.copy(B.color).multiplyScalar(B.intensity),B.castShadow){const te=B.shadow,ue=t.get(B);ue.shadowIntensity=te.intensity,ue.shadowBias=te.bias,ue.shadowNormalBias=te.normalBias,ue.shadowRadius=te.radius,ue.shadowMapSize=te.mapSize,n.directionalShadow[v]=ue,n.directionalShadowMap[v]=oe,n.directionalShadowMatrix[v]=B.shadow.matrix,R++}n.directional[v]=Q,v++}else if(B.isSpotLight){const Q=e.get(B);Q.position.setFromMatrixPosition(B.matrixWorld),Q.color.copy(j).multiplyScalar(K),Q.distance=ie,Q.coneCos=Math.cos(B.angle),Q.penumbraCos=Math.cos(B.angle*(1-B.penumbra)),Q.decay=B.decay,n.spot[b]=Q;const te=B.shadow;if(B.map&&(n.spotLightMap[P]=B.map,P++,te.updateMatrices(B),B.castShadow&&D++),n.spotLightMatrix[b]=te.matrix,B.castShadow){const ue=t.get(B);ue.shadowIntensity=te.intensity,ue.shadowBias=te.bias,ue.shadowNormalBias=te.normalBias,ue.shadowRadius=te.radius,ue.shadowMapSize=te.mapSize,n.spotShadow[b]=ue,n.spotShadowMap[b]=oe,A++}b++}else if(B.isRectAreaLight){const Q=e.get(B);Q.color.copy(j).multiplyScalar(K),Q.halfWidth.set(B.width*.5,0,0),Q.halfHeight.set(0,B.height*.5,0),n.rectArea[g]=Q,g++}else if(B.isPointLight){const Q=e.get(B);if(Q.color.copy(B.color).multiplyScalar(B.intensity),Q.distance=B.distance,Q.decay=B.decay,B.castShadow){const te=B.shadow,ue=t.get(B);ue.shadowIntensity=te.intensity,ue.shadowBias=te.bias,ue.shadowNormalBias=te.normalBias,ue.shadowRadius=te.radius,ue.shadowMapSize=te.mapSize,ue.shadowCameraNear=te.camera.near,ue.shadowCameraFar=te.camera.far,n.pointShadow[y]=ue,n.pointShadowMap[y]=oe,n.pointShadowMatrix[y]=B.shadow.matrix,C++}n.point[y]=Q,y++}else if(B.isHemisphereLight){const Q=e.get(B);Q.skyColor.copy(B.color).multiplyScalar(K),Q.groundColor.copy(B.groundColor).multiplyScalar(K),n.hemi[d]=Q,d++}}g>0&&(i.has("OES_texture_float_linear")===!0?(n.rectAreaLTC1=Pe.LTC_FLOAT_1,n.rectAreaLTC2=Pe.LTC_FLOAT_2):(n.rectAreaLTC1=Pe.LTC_HALF_1,n.rectAreaLTC2=Pe.LTC_HALF_2)),n.ambient[0]=h,n.ambient[1]=p,n.ambient[2]=m;const G=n.hash;(G.directionalLength!==v||G.pointLength!==y||G.spotLength!==b||G.rectAreaLength!==g||G.hemiLength!==d||G.numDirectionalShadows!==R||G.numPointShadows!==C||G.numSpotShadows!==A||G.numSpotMaps!==P||G.numLightProbes!==I)&&(n.directional.length=v,n.spot.length=b,n.rectArea.length=g,n.point.length=y,n.hemi.length=d,n.directionalShadow.length=R,n.directionalShadowMap.length=R,n.pointShadow.length=C,n.pointShadowMap.length=C,n.spotShadow.length=A,n.spotShadowMap.length=A,n.directionalShadowMatrix.length=R,n.pointShadowMatrix.length=C,n.spotLightMatrix.length=A+P-D,n.spotLightMap.length=P,n.numSpotLightShadowsWithMaps=D,n.numLightProbes=I,G.directionalLength=v,G.pointLength=y,G.spotLength=b,G.rectAreaLength=g,G.hemiLength=d,G.numDirectionalShadows=R,G.numPointShadows=C,G.numSpotShadows=A,G.numSpotMaps=P,G.numLightProbes=I,n.version=Jm++)}function u(l,h){let p=0,m=0,v=0,y=0,b=0;const g=h.matrixWorldInverse;for(let d=0,R=l.length;d<R;d++){const C=l[d];if(C.isDirectionalLight){const A=n.directional[p];A.direction.setFromMatrixPosition(C.matrixWorld),r.setFromMatrixPosition(C.target.matrixWorld),A.direction.sub(r),A.direction.transformDirection(g),p++}else if(C.isSpotLight){const A=n.spot[v];A.position.setFromMatrixPosition(C.matrixWorld),A.position.applyMatrix4(g),A.direction.setFromMatrixPosition(C.matrixWorld),r.setFromMatrixPosition(C.target.matrixWorld),A.direction.sub(r),A.direction.transformDirection(g),v++}else if(C.isRectAreaLight){const A=n.rectArea[y];A.position.setFromMatrixPosition(C.matrixWorld),A.position.applyMatrix4(g),a.identity(),s.copy(C.matrixWorld),s.premultiply(g),a.extractRotation(s),A.halfWidth.set(C.width*.5,0,0),A.halfHeight.set(0,C.height*.5,0),A.halfWidth.applyMatrix4(a),A.halfHeight.applyMatrix4(a),y++}else if(C.isPointLight){const A=n.point[m];A.position.setFromMatrixPosition(C.matrixWorld),A.position.applyMatrix4(g),m++}else if(C.isHemisphereLight){const A=n.hemi[b];A.direction.setFromMatrixPosition(C.matrixWorld),A.direction.transformDirection(g),b++}}}return{setup:o,setupView:u,state:n}}function vl(i){const e=new e_(i),t=[],n=[];function r(h){l.camera=h,t.length=0,n.length=0}function s(h){t.push(h)}function a(h){n.push(h)}function o(){e.setup(t)}function u(h){e.setupView(t,h)}const l={lightsArray:t,shadowsArray:n,camera:null,lights:e,transmissionRenderTarget:{}};return{init:r,state:l,setupLights:o,setupLightsView:u,pushLight:s,pushShadow:a}}function t_(i){let e=new WeakMap;function t(r,s=0){const a=e.get(r);let o;return a===void 0?(o=new vl(i),e.set(r,[o])):s>=a.length?(o=new vl(i),a.push(o)):o=a[s],o}function n(){e=new WeakMap}return{get:t,dispose:n}}const n_=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,i_=`uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;
void main() {
	const float samples = float( VSM_SAMPLES );
	float mean = 0.0;
	float squared_mean = 0.0;
	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {
		float uvOffset = uvStart + i * uvStride;
		#ifdef HORIZONTAL_PASS
			vec2 distribution = texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ).rg;
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;
		#else
			float depth = texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ).r;
			mean += depth;
			squared_mean += depth * depth;
		#endif
	}
	mean = mean / samples;
	squared_mean = squared_mean / samples;
	float std_dev = sqrt( max( 0.0, squared_mean - mean * mean ) );
	gl_FragColor = vec4( mean, std_dev, 0.0, 1.0 );
}`,r_=[new $(1,0,0),new $(-1,0,0),new $(0,1,0),new $(0,-1,0),new $(0,0,1),new $(0,0,-1)],s_=[new $(0,-1,0),new $(0,-1,0),new $(0,0,1),new $(0,0,-1),new $(0,-1,0),new $(0,-1,0)],xl=new Ot,cr=new $,Zs=new $;function a_(i,e,t){let n=new ao;const r=new pt,s=new pt,a=new Nt,o=new vf,u=new xf,l={},h=t.maxTextureSize,p={[ai]:on,[on]:ai,[Wn]:Wn},m=new Fn({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new pt},radius:{value:4}},vertexShader:n_,fragmentShader:i_}),v=m.clone();v.defines.HORIZONTAL_PASS=1;const y=new bn;y.setAttribute("position",new _n(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));const b=new In(y,m),g=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=jr;let d=this.type;this.render=function(D,I,G){if(g.enabled===!1||g.autoUpdate===!1&&g.needsUpdate===!1||D.length===0)return;D.type===au&&(et("WebGLShadowMap: PCFSoftShadowMap has been deprecated. Using PCFShadowMap instead."),D.type=jr);const x=i.getRenderTarget(),E=i.getActiveCubeFace(),B=i.getActiveMipmapLevel(),j=i.state;j.setBlending($n),j.buffers.depth.getReversed()===!0?j.buffers.color.setClear(0,0,0,0):j.buffers.color.setClear(1,1,1,1),j.buffers.depth.setTest(!0),j.setScissorTest(!1);const K=d!==this.type;K&&I.traverse(function(ie){ie.material&&(Array.isArray(ie.material)?ie.material.forEach(oe=>oe.needsUpdate=!0):ie.material.needsUpdate=!0)});for(let ie=0,oe=D.length;ie<oe;ie++){const Q=D[ie],te=Q.shadow;if(te===void 0){et("WebGLShadowMap:",Q,"has no shadow.");continue}if(te.autoUpdate===!1&&te.needsUpdate===!1)continue;r.copy(te.mapSize);const ue=te.getFrameExtents();if(r.multiply(ue),s.copy(te.mapSize),(r.x>h||r.y>h)&&(r.x>h&&(s.x=Math.floor(h/ue.x),r.x=s.x*ue.x,te.mapSize.x=s.x),r.y>h&&(s.y=Math.floor(h/ue.y),r.y=s.y*ue.y,te.mapSize.y=s.y)),te.map===null||K===!0){if(te.map!==null&&(te.map.depthTexture!==null&&(te.map.depthTexture.dispose(),te.map.depthTexture=null),te.map.dispose()),this.type===ur){if(Q.isPointLight){et("WebGLShadowMap: VSM shadow maps are not supported for PointLights. Use PCF or BasicShadowMap instead.");continue}te.map=new Dn(r.x,r.y,{format:Wi,type:Yn,minFilter:Qt,magFilter:Qt,generateMipmaps:!1}),te.map.texture.name=Q.name+".shadowMap",te.map.depthTexture=new mr(r.x,r.y,Rn),te.map.depthTexture.name=Q.name+".shadowMapDepth",te.map.depthTexture.format=Kn,te.map.depthTexture.compareFunction=null,te.map.depthTexture.minFilter=Xt,te.map.depthTexture.magFilter=Xt}else{Q.isPointLight?(te.map=new ql(r.x),te.map.depthTexture=new mf(r.x,Ln)):(te.map=new Dn(r.x,r.y),te.map.depthTexture=new mr(r.x,r.y,Ln)),te.map.depthTexture.name=Q.name+".shadowMap",te.map.depthTexture.format=Kn;const Ee=i.state.buffers.depth.getReversed();this.type===jr?(te.map.depthTexture.compareFunction=Ee?io:no,te.map.depthTexture.minFilter=Qt,te.map.depthTexture.magFilter=Qt):(te.map.depthTexture.compareFunction=null,te.map.depthTexture.minFilter=Xt,te.map.depthTexture.magFilter=Xt)}te.camera.updateProjectionMatrix()}const Ae=te.map.isWebGLCubeRenderTarget?6:1;for(let Ee=0;Ee<Ae;Ee++){if(te.map.isWebGLCubeRenderTarget)i.setRenderTarget(te.map,Ee),i.clear();else{Ee===0&&(i.setRenderTarget(te.map),i.clear());const Le=te.getViewport(Ee);a.set(s.x*Le.x,s.y*Le.y,s.x*Le.z,s.y*Le.w),j.viewport(a)}if(Q.isPointLight){const Le=te.camera,at=te.matrix,it=Q.distance||Le.far;it!==Le.far&&(Le.far=it,Le.updateProjectionMatrix()),cr.setFromMatrixPosition(Q.matrixWorld),Le.position.copy(cr),Zs.copy(Le.position),Zs.add(r_[Ee]),Le.up.copy(s_[Ee]),Le.lookAt(Zs),Le.updateMatrixWorld(),at.makeTranslation(-cr.x,-cr.y,-cr.z),xl.multiplyMatrices(Le.projectionMatrix,Le.matrixWorldInverse),te._frustum.setFromProjectionMatrix(xl,Le.coordinateSystem,Le.reversedDepth)}else te.updateMatrices(Q);n=te.getFrustum(),A(I,G,te.camera,Q,this.type)}te.isPointLightShadow!==!0&&this.type===ur&&R(te,G),te.needsUpdate=!1}d=this.type,g.needsUpdate=!1,i.setRenderTarget(x,E,B)};function R(D,I){const G=e.update(b);m.defines.VSM_SAMPLES!==D.blurSamples&&(m.defines.VSM_SAMPLES=D.blurSamples,v.defines.VSM_SAMPLES=D.blurSamples,m.needsUpdate=!0,v.needsUpdate=!0),D.mapPass===null&&(D.mapPass=new Dn(r.x,r.y,{format:Wi,type:Yn})),m.uniforms.shadow_pass.value=D.map.depthTexture,m.uniforms.resolution.value=D.mapSize,m.uniforms.radius.value=D.radius,i.setRenderTarget(D.mapPass),i.clear(),i.renderBufferDirect(I,null,G,m,b,null),v.uniforms.shadow_pass.value=D.mapPass.texture,v.uniforms.resolution.value=D.mapSize,v.uniforms.radius.value=D.radius,i.setRenderTarget(D.map),i.clear(),i.renderBufferDirect(I,null,G,v,b,null)}function C(D,I,G,x){let E=null;const B=G.isPointLight===!0?D.customDistanceMaterial:D.customDepthMaterial;if(B!==void 0)E=B;else if(E=G.isPointLight===!0?u:o,i.localClippingEnabled&&I.clipShadows===!0&&Array.isArray(I.clippingPlanes)&&I.clippingPlanes.length!==0||I.displacementMap&&I.displacementScale!==0||I.alphaMap&&I.alphaTest>0||I.map&&I.alphaTest>0||I.alphaToCoverage===!0){const j=E.uuid,K=I.uuid;let ie=l[j];ie===void 0&&(ie={},l[j]=ie);let oe=ie[K];oe===void 0&&(oe=E.clone(),ie[K]=oe,I.addEventListener("dispose",P)),E=oe}if(E.visible=I.visible,E.wireframe=I.wireframe,x===ur?E.side=I.shadowSide!==null?I.shadowSide:I.side:E.side=I.shadowSide!==null?I.shadowSide:p[I.side],E.alphaMap=I.alphaMap,E.alphaTest=I.alphaToCoverage===!0?.5:I.alphaTest,E.map=I.map,E.clipShadows=I.clipShadows,E.clippingPlanes=I.clippingPlanes,E.clipIntersection=I.clipIntersection,E.displacementMap=I.displacementMap,E.displacementScale=I.displacementScale,E.displacementBias=I.displacementBias,E.wireframeLinewidth=I.wireframeLinewidth,E.linewidth=I.linewidth,G.isPointLight===!0&&E.isMeshDistanceMaterial===!0){const j=i.properties.get(E);j.light=G}return E}function A(D,I,G,x,E){if(D.visible===!1)return;if(D.layers.test(I.layers)&&(D.isMesh||D.isLine||D.isPoints)&&(D.castShadow||D.receiveShadow&&E===ur)&&(!D.frustumCulled||n.intersectsObject(D))){D.modelViewMatrix.multiplyMatrices(G.matrixWorldInverse,D.matrixWorld);const K=e.update(D),ie=D.material;if(Array.isArray(ie)){const oe=K.groups;for(let Q=0,te=oe.length;Q<te;Q++){const ue=oe[Q],Ae=ie[ue.materialIndex];if(Ae&&Ae.visible){const Ee=C(D,Ae,x,E);D.onBeforeShadow(i,D,I,G,K,Ee,ue),i.renderBufferDirect(G,null,K,Ee,D,ue),D.onAfterShadow(i,D,I,G,K,Ee,ue)}}}else if(ie.visible){const oe=C(D,ie,x,E);D.onBeforeShadow(i,D,I,G,K,oe,null),i.renderBufferDirect(G,null,K,oe,D,null),D.onAfterShadow(i,D,I,G,K,oe,null)}}const j=D.children;for(let K=0,ie=j.length;K<ie;K++)A(j[K],I,G,x,E)}function P(D){D.target.removeEventListener("dispose",P);for(const G in l){const x=l[G],E=D.target.uuid;E in x&&(x[E].dispose(),delete x[E])}}}const o_={[ea]:ta,[na]:sa,[ia]:aa,[Hi]:ra,[ta]:ea,[sa]:na,[aa]:ia,[ra]:Hi};function l_(i,e){function t(){let O=!1;const we=new Nt;let _e=null;const De=new Nt(0,0,0,0);return{setMask:function(de){_e!==de&&!O&&(i.colorMask(de,de,de,de),_e=de)},setLocked:function(de){O=de},setClear:function(de,le,xe,Ke,At){At===!0&&(de*=Ke,le*=Ke,xe*=Ke),we.set(de,le,xe,Ke),De.equals(we)===!1&&(i.clearColor(de,le,xe,Ke),De.copy(we))},reset:function(){O=!1,_e=null,De.set(-1,0,0,0)}}}function n(){let O=!1,we=!1,_e=null,De=null,de=null;return{setReversed:function(le){if(we!==le){const xe=e.get("EXT_clip_control");le?xe.clipControlEXT(xe.LOWER_LEFT_EXT,xe.ZERO_TO_ONE_EXT):xe.clipControlEXT(xe.LOWER_LEFT_EXT,xe.NEGATIVE_ONE_TO_ONE_EXT),we=le;const Ke=de;de=null,this.setClear(Ke)}},getReversed:function(){return we},setTest:function(le){le?fe(i.DEPTH_TEST):Ne(i.DEPTH_TEST)},setMask:function(le){_e!==le&&!O&&(i.depthMask(le),_e=le)},setFunc:function(le){if(we&&(le=o_[le]),De!==le){switch(le){case ea:i.depthFunc(i.NEVER);break;case ta:i.depthFunc(i.ALWAYS);break;case na:i.depthFunc(i.LESS);break;case Hi:i.depthFunc(i.LEQUAL);break;case ia:i.depthFunc(i.EQUAL);break;case ra:i.depthFunc(i.GEQUAL);break;case sa:i.depthFunc(i.GREATER);break;case aa:i.depthFunc(i.NOTEQUAL);break;default:i.depthFunc(i.LEQUAL)}De=le}},setLocked:function(le){O=le},setClear:function(le){de!==le&&(we&&(le=1-le),i.clearDepth(le),de=le)},reset:function(){O=!1,_e=null,De=null,de=null,we=!1}}}function r(){let O=!1,we=null,_e=null,De=null,de=null,le=null,xe=null,Ke=null,At=null;return{setTest:function(vt){O||(vt?fe(i.STENCIL_TEST):Ne(i.STENCIL_TEST))},setMask:function(vt){we!==vt&&!O&&(i.stencilMask(vt),we=vt)},setFunc:function(vt,$t,vn){(_e!==vt||De!==$t||de!==vn)&&(i.stencilFunc(vt,$t,vn),_e=vt,De=$t,de=vn)},setOp:function(vt,$t,vn){(le!==vt||xe!==$t||Ke!==vn)&&(i.stencilOp(vt,$t,vn),le=vt,xe=$t,Ke=vn)},setLocked:function(vt){O=vt},setClear:function(vt){At!==vt&&(i.clearStencil(vt),At=vt)},reset:function(){O=!1,we=null,_e=null,De=null,de=null,le=null,xe=null,Ke=null,At=null}}}const s=new t,a=new n,o=new r,u=new WeakMap,l=new WeakMap;let h={},p={},m=new WeakMap,v=[],y=null,b=!1,g=null,d=null,R=null,C=null,A=null,P=null,D=null,I=new yt(0,0,0),G=0,x=!1,E=null,B=null,j=null,K=null,ie=null;const oe=i.getParameter(i.MAX_COMBINED_TEXTURE_IMAGE_UNITS);let Q=!1,te=0;const ue=i.getParameter(i.VERSION);ue.indexOf("WebGL")!==-1?(te=parseFloat(/^WebGL (\d)/.exec(ue)[1]),Q=te>=1):ue.indexOf("OpenGL ES")!==-1&&(te=parseFloat(/^OpenGL ES (\d)/.exec(ue)[1]),Q=te>=2);let Ae=null,Ee={};const Le=i.getParameter(i.SCISSOR_BOX),at=i.getParameter(i.VIEWPORT),it=new Nt().fromArray(Le),Ut=new Nt().fromArray(at);function Dt(O,we,_e,De){const de=new Uint8Array(4),le=i.createTexture();i.bindTexture(O,le),i.texParameteri(O,i.TEXTURE_MIN_FILTER,i.NEAREST),i.texParameteri(O,i.TEXTURE_MAG_FILTER,i.NEAREST);for(let xe=0;xe<_e;xe++)O===i.TEXTURE_3D||O===i.TEXTURE_2D_ARRAY?i.texImage3D(we,0,i.RGBA,1,1,De,0,i.RGBA,i.UNSIGNED_BYTE,de):i.texImage2D(we+xe,0,i.RGBA,1,1,0,i.RGBA,i.UNSIGNED_BYTE,de);return le}const re={};re[i.TEXTURE_2D]=Dt(i.TEXTURE_2D,i.TEXTURE_2D,1),re[i.TEXTURE_CUBE_MAP]=Dt(i.TEXTURE_CUBE_MAP,i.TEXTURE_CUBE_MAP_POSITIVE_X,6),re[i.TEXTURE_2D_ARRAY]=Dt(i.TEXTURE_2D_ARRAY,i.TEXTURE_2D_ARRAY,1,1),re[i.TEXTURE_3D]=Dt(i.TEXTURE_3D,i.TEXTURE_3D,1,1),s.setClear(0,0,0,1),a.setClear(1),o.setClear(0),fe(i.DEPTH_TEST),a.setFunc(Hi),nt(!1),Ft(yo),fe(i.CULL_FACE),ut($n);function fe(O){h[O]!==!0&&(i.enable(O),h[O]=!0)}function Ne(O){h[O]!==!1&&(i.disable(O),h[O]=!1)}function Qe(O,we){return p[O]!==we?(i.bindFramebuffer(O,we),p[O]=we,O===i.DRAW_FRAMEBUFFER&&(p[i.FRAMEBUFFER]=we),O===i.FRAMEBUFFER&&(p[i.DRAW_FRAMEBUFFER]=we),!0):!1}function Ve(O,we){let _e=v,De=!1;if(O){_e=m.get(we),_e===void 0&&(_e=[],m.set(we,_e));const de=O.textures;if(_e.length!==de.length||_e[0]!==i.COLOR_ATTACHMENT0){for(let le=0,xe=de.length;le<xe;le++)_e[le]=i.COLOR_ATTACHMENT0+le;_e.length=de.length,De=!0}}else _e[0]!==i.BACK&&(_e[0]=i.BACK,De=!0);De&&i.drawBuffers(_e)}function ht(O){return y!==O?(i.useProgram(O),y=O,!0):!1}const Vt={[gi]:i.FUNC_ADD,[lu]:i.FUNC_SUBTRACT,[cu]:i.FUNC_REVERSE_SUBTRACT};Vt[uu]=i.MIN,Vt[fu]=i.MAX;const Je={[hu]:i.ZERO,[du]:i.ONE,[pu]:i.SRC_COLOR,[Js]:i.SRC_ALPHA,[Mu]:i.SRC_ALPHA_SATURATE,[vu]:i.DST_COLOR,[_u]:i.DST_ALPHA,[mu]:i.ONE_MINUS_SRC_COLOR,[Qs]:i.ONE_MINUS_SRC_ALPHA,[xu]:i.ONE_MINUS_DST_COLOR,[gu]:i.ONE_MINUS_DST_ALPHA,[Su]:i.CONSTANT_COLOR,[yu]:i.ONE_MINUS_CONSTANT_COLOR,[Eu]:i.CONSTANT_ALPHA,[bu]:i.ONE_MINUS_CONSTANT_ALPHA};function ut(O,we,_e,De,de,le,xe,Ke,At,vt){if(O===$n){b===!0&&(Ne(i.BLEND),b=!1);return}if(b===!1&&(fe(i.BLEND),b=!0),O!==ou){if(O!==g||vt!==x){if((d!==gi||A!==gi)&&(i.blendEquation(i.FUNC_ADD),d=gi,A=gi),vt)switch(O){case zi:i.blendFuncSeparate(i.ONE,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case Eo:i.blendFunc(i.ONE,i.ONE);break;case bo:i.blendFuncSeparate(i.ZERO,i.ONE_MINUS_SRC_COLOR,i.ZERO,i.ONE);break;case To:i.blendFuncSeparate(i.DST_COLOR,i.ONE_MINUS_SRC_ALPHA,i.ZERO,i.ONE);break;default:xt("WebGLState: Invalid blending: ",O);break}else switch(O){case zi:i.blendFuncSeparate(i.SRC_ALPHA,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case Eo:i.blendFuncSeparate(i.SRC_ALPHA,i.ONE,i.ONE,i.ONE);break;case bo:xt("WebGLState: SubtractiveBlending requires material.premultipliedAlpha = true");break;case To:xt("WebGLState: MultiplyBlending requires material.premultipliedAlpha = true");break;default:xt("WebGLState: Invalid blending: ",O);break}R=null,C=null,P=null,D=null,I.set(0,0,0),G=0,g=O,x=vt}return}de=de||we,le=le||_e,xe=xe||De,(we!==d||de!==A)&&(i.blendEquationSeparate(Vt[we],Vt[de]),d=we,A=de),(_e!==R||De!==C||le!==P||xe!==D)&&(i.blendFuncSeparate(Je[_e],Je[De],Je[le],Je[xe]),R=_e,C=De,P=le,D=xe),(Ke.equals(I)===!1||At!==G)&&(i.blendColor(Ke.r,Ke.g,Ke.b,At),I.copy(Ke),G=At),g=O,x=!1}function Tt(O,we){O.side===Wn?Ne(i.CULL_FACE):fe(i.CULL_FACE);let _e=O.side===on;we&&(_e=!_e),nt(_e),O.blending===zi&&O.transparent===!1?ut($n):ut(O.blending,O.blendEquation,O.blendSrc,O.blendDst,O.blendEquationAlpha,O.blendSrcAlpha,O.blendDstAlpha,O.blendColor,O.blendAlpha,O.premultipliedAlpha),a.setFunc(O.depthFunc),a.setTest(O.depthTest),a.setMask(O.depthWrite),s.setMask(O.colorWrite);const De=O.stencilWrite;o.setTest(De),De&&(o.setMask(O.stencilWriteMask),o.setFunc(O.stencilFunc,O.stencilRef,O.stencilFuncMask),o.setOp(O.stencilFail,O.stencilZFail,O.stencilZPass)),It(O.polygonOffset,O.polygonOffsetFactor,O.polygonOffsetUnits),O.alphaToCoverage===!0?fe(i.SAMPLE_ALPHA_TO_COVERAGE):Ne(i.SAMPLE_ALPHA_TO_COVERAGE)}function nt(O){E!==O&&(O?i.frontFace(i.CW):i.frontFace(i.CCW),E=O)}function Ft(O){O!==ru?(fe(i.CULL_FACE),O!==B&&(O===yo?i.cullFace(i.BACK):O===su?i.cullFace(i.FRONT):i.cullFace(i.FRONT_AND_BACK))):Ne(i.CULL_FACE),B=O}function F(O){O!==j&&(Q&&i.lineWidth(O),j=O)}function It(O,we,_e){O?(fe(i.POLYGON_OFFSET_FILL),(K!==we||ie!==_e)&&(i.polygonOffset(we,_e),K=we,ie=_e)):Ne(i.POLYGON_OFFSET_FILL)}function mt(O){O?fe(i.SCISSOR_TEST):Ne(i.SCISSOR_TEST)}function gt(O){O===void 0&&(O=i.TEXTURE0+oe-1),Ae!==O&&(i.activeTexture(O),Ae=O)}function Oe(O,we,_e){_e===void 0&&(Ae===null?_e=i.TEXTURE0+oe-1:_e=Ae);let De=Ee[_e];De===void 0&&(De={type:void 0,texture:void 0},Ee[_e]=De),(De.type!==O||De.texture!==we)&&(Ae!==_e&&(i.activeTexture(_e),Ae=_e),i.bindTexture(O,we||re[O]),De.type=O,De.texture=we)}function w(){const O=Ee[Ae];O!==void 0&&O.type!==void 0&&(i.bindTexture(O.type,null),O.type=void 0,O.texture=void 0)}function _(){try{i.compressedTexImage2D(...arguments)}catch(O){xt("WebGLState:",O)}}function V(){try{i.compressedTexImage3D(...arguments)}catch(O){xt("WebGLState:",O)}}function ne(){try{i.texSubImage2D(...arguments)}catch(O){xt("WebGLState:",O)}}function se(){try{i.texSubImage3D(...arguments)}catch(O){xt("WebGLState:",O)}}function ee(){try{i.compressedTexSubImage2D(...arguments)}catch(O){xt("WebGLState:",O)}}function Me(){try{i.compressedTexSubImage3D(...arguments)}catch(O){xt("WebGLState:",O)}}function ve(){try{i.texStorage2D(...arguments)}catch(O){xt("WebGLState:",O)}}function me(){try{i.texStorage3D(...arguments)}catch(O){xt("WebGLState:",O)}}function Ye(){try{i.texImage2D(...arguments)}catch(O){xt("WebGLState:",O)}}function ce(){try{i.texImage3D(...arguments)}catch(O){xt("WebGLState:",O)}}function Se(O){it.equals(O)===!1&&(i.scissor(O.x,O.y,O.z,O.w),it.copy(O))}function Be(O){Ut.equals(O)===!1&&(i.viewport(O.x,O.y,O.z,O.w),Ut.copy(O))}function Ge(O,we){let _e=l.get(we);_e===void 0&&(_e=new WeakMap,l.set(we,_e));let De=_e.get(O);De===void 0&&(De=i.getUniformBlockIndex(we,O.name),_e.set(O,De))}function ye(O,we){const De=l.get(we).get(O);u.get(we)!==De&&(i.uniformBlockBinding(we,De,O.__bindingPointIndex),u.set(we,De))}function tt(){i.disable(i.BLEND),i.disable(i.CULL_FACE),i.disable(i.DEPTH_TEST),i.disable(i.POLYGON_OFFSET_FILL),i.disable(i.SCISSOR_TEST),i.disable(i.STENCIL_TEST),i.disable(i.SAMPLE_ALPHA_TO_COVERAGE),i.blendEquation(i.FUNC_ADD),i.blendFunc(i.ONE,i.ZERO),i.blendFuncSeparate(i.ONE,i.ZERO,i.ONE,i.ZERO),i.blendColor(0,0,0,0),i.colorMask(!0,!0,!0,!0),i.clearColor(0,0,0,0),i.depthMask(!0),i.depthFunc(i.LESS),a.setReversed(!1),i.clearDepth(1),i.stencilMask(4294967295),i.stencilFunc(i.ALWAYS,0,4294967295),i.stencilOp(i.KEEP,i.KEEP,i.KEEP),i.clearStencil(0),i.cullFace(i.BACK),i.frontFace(i.CCW),i.polygonOffset(0,0),i.activeTexture(i.TEXTURE0),i.bindFramebuffer(i.FRAMEBUFFER,null),i.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),i.bindFramebuffer(i.READ_FRAMEBUFFER,null),i.useProgram(null),i.lineWidth(1),i.scissor(0,0,i.canvas.width,i.canvas.height),i.viewport(0,0,i.canvas.width,i.canvas.height),h={},Ae=null,Ee={},p={},m=new WeakMap,v=[],y=null,b=!1,g=null,d=null,R=null,C=null,A=null,P=null,D=null,I=new yt(0,0,0),G=0,x=!1,E=null,B=null,j=null,K=null,ie=null,it.set(0,0,i.canvas.width,i.canvas.height),Ut.set(0,0,i.canvas.width,i.canvas.height),s.reset(),a.reset(),o.reset()}return{buffers:{color:s,depth:a,stencil:o},enable:fe,disable:Ne,bindFramebuffer:Qe,drawBuffers:Ve,useProgram:ht,setBlending:ut,setMaterial:Tt,setFlipSided:nt,setCullFace:Ft,setLineWidth:F,setPolygonOffset:It,setScissorTest:mt,activeTexture:gt,bindTexture:Oe,unbindTexture:w,compressedTexImage2D:_,compressedTexImage3D:V,texImage2D:Ye,texImage3D:ce,updateUBOMapping:Ge,uniformBlockBinding:ye,texStorage2D:ve,texStorage3D:me,texSubImage2D:ne,texSubImage3D:se,compressedTexSubImage2D:ee,compressedTexSubImage3D:Me,scissor:Se,viewport:Be,reset:tt}}function c_(i,e,t,n,r,s,a){const o=e.has("WEBGL_multisampled_render_to_texture")?e.get("WEBGL_multisampled_render_to_texture"):null,u=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),l=new pt,h=new WeakMap;let p;const m=new WeakMap;let v=!1;try{v=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function y(w,_){return v?new OffscreenCanvas(w,_):rs("canvas")}function b(w,_,V){let ne=1;const se=Oe(w);if((se.width>V||se.height>V)&&(ne=V/Math.max(se.width,se.height)),ne<1)if(typeof HTMLImageElement<"u"&&w instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&w instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&w instanceof ImageBitmap||typeof VideoFrame<"u"&&w instanceof VideoFrame){const ee=Math.floor(ne*se.width),Me=Math.floor(ne*se.height);p===void 0&&(p=y(ee,Me));const ve=_?y(ee,Me):p;return ve.width=ee,ve.height=Me,ve.getContext("2d").drawImage(w,0,0,ee,Me),et("WebGLRenderer: Texture has been resized from ("+se.width+"x"+se.height+") to ("+ee+"x"+Me+")."),ve}else return"data"in w&&et("WebGLRenderer: Image in DataTexture is too big ("+se.width+"x"+se.height+")."),w;return w}function g(w){return w.generateMipmaps}function d(w){i.generateMipmap(w)}function R(w){return w.isWebGLCubeRenderTarget?i.TEXTURE_CUBE_MAP:w.isWebGL3DRenderTarget?i.TEXTURE_3D:w.isWebGLArrayRenderTarget||w.isCompressedArrayTexture?i.TEXTURE_2D_ARRAY:i.TEXTURE_2D}function C(w,_,V,ne,se=!1){if(w!==null){if(i[w]!==void 0)return i[w];et("WebGLRenderer: Attempt to use non-existing WebGL internal format '"+w+"'")}let ee=_;if(_===i.RED&&(V===i.FLOAT&&(ee=i.R32F),V===i.HALF_FLOAT&&(ee=i.R16F),V===i.UNSIGNED_BYTE&&(ee=i.R8)),_===i.RED_INTEGER&&(V===i.UNSIGNED_BYTE&&(ee=i.R8UI),V===i.UNSIGNED_SHORT&&(ee=i.R16UI),V===i.UNSIGNED_INT&&(ee=i.R32UI),V===i.BYTE&&(ee=i.R8I),V===i.SHORT&&(ee=i.R16I),V===i.INT&&(ee=i.R32I)),_===i.RG&&(V===i.FLOAT&&(ee=i.RG32F),V===i.HALF_FLOAT&&(ee=i.RG16F),V===i.UNSIGNED_BYTE&&(ee=i.RG8)),_===i.RG_INTEGER&&(V===i.UNSIGNED_BYTE&&(ee=i.RG8UI),V===i.UNSIGNED_SHORT&&(ee=i.RG16UI),V===i.UNSIGNED_INT&&(ee=i.RG32UI),V===i.BYTE&&(ee=i.RG8I),V===i.SHORT&&(ee=i.RG16I),V===i.INT&&(ee=i.RG32I)),_===i.RGB_INTEGER&&(V===i.UNSIGNED_BYTE&&(ee=i.RGB8UI),V===i.UNSIGNED_SHORT&&(ee=i.RGB16UI),V===i.UNSIGNED_INT&&(ee=i.RGB32UI),V===i.BYTE&&(ee=i.RGB8I),V===i.SHORT&&(ee=i.RGB16I),V===i.INT&&(ee=i.RGB32I)),_===i.RGBA_INTEGER&&(V===i.UNSIGNED_BYTE&&(ee=i.RGBA8UI),V===i.UNSIGNED_SHORT&&(ee=i.RGBA16UI),V===i.UNSIGNED_INT&&(ee=i.RGBA32UI),V===i.BYTE&&(ee=i.RGBA8I),V===i.SHORT&&(ee=i.RGBA16I),V===i.INT&&(ee=i.RGBA32I)),_===i.RGB&&(V===i.UNSIGNED_INT_5_9_9_9_REV&&(ee=i.RGB9_E5),V===i.UNSIGNED_INT_10F_11F_11F_REV&&(ee=i.R11F_G11F_B10F)),_===i.RGBA){const Me=se?ns:_t.getTransfer(ne);V===i.FLOAT&&(ee=i.RGBA32F),V===i.HALF_FLOAT&&(ee=i.RGBA16F),V===i.UNSIGNED_BYTE&&(ee=Me===wt?i.SRGB8_ALPHA8:i.RGBA8),V===i.UNSIGNED_SHORT_4_4_4_4&&(ee=i.RGBA4),V===i.UNSIGNED_SHORT_5_5_5_1&&(ee=i.RGB5_A1)}return(ee===i.R16F||ee===i.R32F||ee===i.RG16F||ee===i.RG32F||ee===i.RGBA16F||ee===i.RGBA32F)&&e.get("EXT_color_buffer_float"),ee}function A(w,_){let V;return w?_===null||_===Ln||_===dr?V=i.DEPTH24_STENCIL8:_===Rn?V=i.DEPTH32F_STENCIL8:_===hr&&(V=i.DEPTH24_STENCIL8,et("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")):_===null||_===Ln||_===dr?V=i.DEPTH_COMPONENT24:_===Rn?V=i.DEPTH_COMPONENT32F:_===hr&&(V=i.DEPTH_COMPONENT16),V}function P(w,_){return g(w)===!0||w.isFramebufferTexture&&w.minFilter!==Xt&&w.minFilter!==Qt?Math.log2(Math.max(_.width,_.height))+1:w.mipmaps!==void 0&&w.mipmaps.length>0?w.mipmaps.length:w.isCompressedTexture&&Array.isArray(w.image)?_.mipmaps.length:1}function D(w){const _=w.target;_.removeEventListener("dispose",D),G(_),_.isVideoTexture&&h.delete(_)}function I(w){const _=w.target;_.removeEventListener("dispose",I),E(_)}function G(w){const _=n.get(w);if(_.__webglInit===void 0)return;const V=w.source,ne=m.get(V);if(ne){const se=ne[_.__cacheKey];se.usedTimes--,se.usedTimes===0&&x(w),Object.keys(ne).length===0&&m.delete(V)}n.remove(w)}function x(w){const _=n.get(w);i.deleteTexture(_.__webglTexture);const V=w.source,ne=m.get(V);delete ne[_.__cacheKey],a.memory.textures--}function E(w){const _=n.get(w);if(w.depthTexture&&(w.depthTexture.dispose(),n.remove(w.depthTexture)),w.isWebGLCubeRenderTarget)for(let ne=0;ne<6;ne++){if(Array.isArray(_.__webglFramebuffer[ne]))for(let se=0;se<_.__webglFramebuffer[ne].length;se++)i.deleteFramebuffer(_.__webglFramebuffer[ne][se]);else i.deleteFramebuffer(_.__webglFramebuffer[ne]);_.__webglDepthbuffer&&i.deleteRenderbuffer(_.__webglDepthbuffer[ne])}else{if(Array.isArray(_.__webglFramebuffer))for(let ne=0;ne<_.__webglFramebuffer.length;ne++)i.deleteFramebuffer(_.__webglFramebuffer[ne]);else i.deleteFramebuffer(_.__webglFramebuffer);if(_.__webglDepthbuffer&&i.deleteRenderbuffer(_.__webglDepthbuffer),_.__webglMultisampledFramebuffer&&i.deleteFramebuffer(_.__webglMultisampledFramebuffer),_.__webglColorRenderbuffer)for(let ne=0;ne<_.__webglColorRenderbuffer.length;ne++)_.__webglColorRenderbuffer[ne]&&i.deleteRenderbuffer(_.__webglColorRenderbuffer[ne]);_.__webglDepthRenderbuffer&&i.deleteRenderbuffer(_.__webglDepthRenderbuffer)}const V=w.textures;for(let ne=0,se=V.length;ne<se;ne++){const ee=n.get(V[ne]);ee.__webglTexture&&(i.deleteTexture(ee.__webglTexture),a.memory.textures--),n.remove(V[ne])}n.remove(w)}let B=0;function j(){B=0}function K(){const w=B;return w>=r.maxTextures&&et("WebGLTextures: Trying to use "+w+" texture units while this GPU supports only "+r.maxTextures),B+=1,w}function ie(w){const _=[];return _.push(w.wrapS),_.push(w.wrapT),_.push(w.wrapR||0),_.push(w.magFilter),_.push(w.minFilter),_.push(w.anisotropy),_.push(w.internalFormat),_.push(w.format),_.push(w.type),_.push(w.generateMipmaps),_.push(w.premultiplyAlpha),_.push(w.flipY),_.push(w.unpackAlignment),_.push(w.colorSpace),_.join()}function oe(w,_){const V=n.get(w);if(w.isVideoTexture&&mt(w),w.isRenderTargetTexture===!1&&w.isExternalTexture!==!0&&w.version>0&&V.__version!==w.version){const ne=w.image;if(ne===null)et("WebGLRenderer: Texture marked for update but no image data found.");else if(ne.complete===!1)et("WebGLRenderer: Texture marked for update but image is incomplete");else{re(V,w,_);return}}else w.isExternalTexture&&(V.__webglTexture=w.sourceTexture?w.sourceTexture:null);t.bindTexture(i.TEXTURE_2D,V.__webglTexture,i.TEXTURE0+_)}function Q(w,_){const V=n.get(w);if(w.isRenderTargetTexture===!1&&w.version>0&&V.__version!==w.version){re(V,w,_);return}else w.isExternalTexture&&(V.__webglTexture=w.sourceTexture?w.sourceTexture:null);t.bindTexture(i.TEXTURE_2D_ARRAY,V.__webglTexture,i.TEXTURE0+_)}function te(w,_){const V=n.get(w);if(w.isRenderTargetTexture===!1&&w.version>0&&V.__version!==w.version){re(V,w,_);return}t.bindTexture(i.TEXTURE_3D,V.__webglTexture,i.TEXTURE0+_)}function ue(w,_){const V=n.get(w);if(w.isCubeDepthTexture!==!0&&w.version>0&&V.__version!==w.version){fe(V,w,_);return}t.bindTexture(i.TEXTURE_CUBE_MAP,V.__webglTexture,i.TEXTURE0+_)}const Ae={[ca]:i.REPEAT,[Xn]:i.CLAMP_TO_EDGE,[ua]:i.MIRRORED_REPEAT},Ee={[Xt]:i.NEAREST,[wu]:i.NEAREST_MIPMAP_NEAREST,[Cr]:i.NEAREST_MIPMAP_LINEAR,[Qt]:i.LINEAR,[Ms]:i.LINEAR_MIPMAP_NEAREST,[xi]:i.LINEAR_MIPMAP_LINEAR},Le={[Pu]:i.NEVER,[Fu]:i.ALWAYS,[Du]:i.LESS,[no]:i.LEQUAL,[Lu]:i.EQUAL,[io]:i.GEQUAL,[Uu]:i.GREATER,[Iu]:i.NOTEQUAL};function at(w,_){if(_.type===Rn&&e.has("OES_texture_float_linear")===!1&&(_.magFilter===Qt||_.magFilter===Ms||_.magFilter===Cr||_.magFilter===xi||_.minFilter===Qt||_.minFilter===Ms||_.minFilter===Cr||_.minFilter===xi)&&et("WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."),i.texParameteri(w,i.TEXTURE_WRAP_S,Ae[_.wrapS]),i.texParameteri(w,i.TEXTURE_WRAP_T,Ae[_.wrapT]),(w===i.TEXTURE_3D||w===i.TEXTURE_2D_ARRAY)&&i.texParameteri(w,i.TEXTURE_WRAP_R,Ae[_.wrapR]),i.texParameteri(w,i.TEXTURE_MAG_FILTER,Ee[_.magFilter]),i.texParameteri(w,i.TEXTURE_MIN_FILTER,Ee[_.minFilter]),_.compareFunction&&(i.texParameteri(w,i.TEXTURE_COMPARE_MODE,i.COMPARE_REF_TO_TEXTURE),i.texParameteri(w,i.TEXTURE_COMPARE_FUNC,Le[_.compareFunction])),e.has("EXT_texture_filter_anisotropic")===!0){if(_.magFilter===Xt||_.minFilter!==Cr&&_.minFilter!==xi||_.type===Rn&&e.has("OES_texture_float_linear")===!1)return;if(_.anisotropy>1||n.get(_).__currentAnisotropy){const V=e.get("EXT_texture_filter_anisotropic");i.texParameterf(w,V.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(_.anisotropy,r.getMaxAnisotropy())),n.get(_).__currentAnisotropy=_.anisotropy}}}function it(w,_){let V=!1;w.__webglInit===void 0&&(w.__webglInit=!0,_.addEventListener("dispose",D));const ne=_.source;let se=m.get(ne);se===void 0&&(se={},m.set(ne,se));const ee=ie(_);if(ee!==w.__cacheKey){se[ee]===void 0&&(se[ee]={texture:i.createTexture(),usedTimes:0},a.memory.textures++,V=!0),se[ee].usedTimes++;const Me=se[w.__cacheKey];Me!==void 0&&(se[w.__cacheKey].usedTimes--,Me.usedTimes===0&&x(_)),w.__cacheKey=ee,w.__webglTexture=se[ee].texture}return V}function Ut(w,_,V){return Math.floor(Math.floor(w/V)/_)}function Dt(w,_,V,ne){const ee=w.updateRanges;if(ee.length===0)t.texSubImage2D(i.TEXTURE_2D,0,0,0,_.width,_.height,V,ne,_.data);else{ee.sort((ce,Se)=>ce.start-Se.start);let Me=0;for(let ce=1;ce<ee.length;ce++){const Se=ee[Me],Be=ee[ce],Ge=Se.start+Se.count,ye=Ut(Be.start,_.width,4),tt=Ut(Se.start,_.width,4);Be.start<=Ge+1&&ye===tt&&Ut(Be.start+Be.count-1,_.width,4)===ye?Se.count=Math.max(Se.count,Be.start+Be.count-Se.start):(++Me,ee[Me]=Be)}ee.length=Me+1;const ve=i.getParameter(i.UNPACK_ROW_LENGTH),me=i.getParameter(i.UNPACK_SKIP_PIXELS),Ye=i.getParameter(i.UNPACK_SKIP_ROWS);i.pixelStorei(i.UNPACK_ROW_LENGTH,_.width);for(let ce=0,Se=ee.length;ce<Se;ce++){const Be=ee[ce],Ge=Math.floor(Be.start/4),ye=Math.ceil(Be.count/4),tt=Ge%_.width,O=Math.floor(Ge/_.width),we=ye,_e=1;i.pixelStorei(i.UNPACK_SKIP_PIXELS,tt),i.pixelStorei(i.UNPACK_SKIP_ROWS,O),t.texSubImage2D(i.TEXTURE_2D,0,tt,O,we,_e,V,ne,_.data)}w.clearUpdateRanges(),i.pixelStorei(i.UNPACK_ROW_LENGTH,ve),i.pixelStorei(i.UNPACK_SKIP_PIXELS,me),i.pixelStorei(i.UNPACK_SKIP_ROWS,Ye)}}function re(w,_,V){let ne=i.TEXTURE_2D;(_.isDataArrayTexture||_.isCompressedArrayTexture)&&(ne=i.TEXTURE_2D_ARRAY),_.isData3DTexture&&(ne=i.TEXTURE_3D);const se=it(w,_),ee=_.source;t.bindTexture(ne,w.__webglTexture,i.TEXTURE0+V);const Me=n.get(ee);if(ee.version!==Me.__version||se===!0){t.activeTexture(i.TEXTURE0+V);const ve=_t.getPrimaries(_t.workingColorSpace),me=_.colorSpace===ri?null:_t.getPrimaries(_.colorSpace),Ye=_.colorSpace===ri||ve===me?i.NONE:i.BROWSER_DEFAULT_WEBGL;i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,_.flipY),i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,_.premultiplyAlpha),i.pixelStorei(i.UNPACK_ALIGNMENT,_.unpackAlignment),i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,Ye);let ce=b(_.image,!1,r.maxTextureSize);ce=gt(_,ce);const Se=s.convert(_.format,_.colorSpace),Be=s.convert(_.type);let Ge=C(_.internalFormat,Se,Be,_.colorSpace,_.isVideoTexture);at(ne,_);let ye;const tt=_.mipmaps,O=_.isVideoTexture!==!0,we=Me.__version===void 0||se===!0,_e=ee.dataReady,De=P(_,ce);if(_.isDepthTexture)Ge=A(_.format===Mi,_.type),we&&(O?t.texStorage2D(i.TEXTURE_2D,1,Ge,ce.width,ce.height):t.texImage2D(i.TEXTURE_2D,0,Ge,ce.width,ce.height,0,Se,Be,null));else if(_.isDataTexture)if(tt.length>0){O&&we&&t.texStorage2D(i.TEXTURE_2D,De,Ge,tt[0].width,tt[0].height);for(let de=0,le=tt.length;de<le;de++)ye=tt[de],O?_e&&t.texSubImage2D(i.TEXTURE_2D,de,0,0,ye.width,ye.height,Se,Be,ye.data):t.texImage2D(i.TEXTURE_2D,de,Ge,ye.width,ye.height,0,Se,Be,ye.data);_.generateMipmaps=!1}else O?(we&&t.texStorage2D(i.TEXTURE_2D,De,Ge,ce.width,ce.height),_e&&Dt(_,ce,Se,Be)):t.texImage2D(i.TEXTURE_2D,0,Ge,ce.width,ce.height,0,Se,Be,ce.data);else if(_.isCompressedTexture)if(_.isCompressedArrayTexture){O&&we&&t.texStorage3D(i.TEXTURE_2D_ARRAY,De,Ge,tt[0].width,tt[0].height,ce.depth);for(let de=0,le=tt.length;de<le;de++)if(ye=tt[de],_.format!==En)if(Se!==null)if(O){if(_e)if(_.layerUpdates.size>0){const xe=jo(ye.width,ye.height,_.format,_.type);for(const Ke of _.layerUpdates){const At=ye.data.subarray(Ke*xe/ye.data.BYTES_PER_ELEMENT,(Ke+1)*xe/ye.data.BYTES_PER_ELEMENT);t.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,de,0,0,Ke,ye.width,ye.height,1,Se,At)}_.clearLayerUpdates()}else t.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,de,0,0,0,ye.width,ye.height,ce.depth,Se,ye.data)}else t.compressedTexImage3D(i.TEXTURE_2D_ARRAY,de,Ge,ye.width,ye.height,ce.depth,0,ye.data,0,0);else et("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");else O?_e&&t.texSubImage3D(i.TEXTURE_2D_ARRAY,de,0,0,0,ye.width,ye.height,ce.depth,Se,Be,ye.data):t.texImage3D(i.TEXTURE_2D_ARRAY,de,Ge,ye.width,ye.height,ce.depth,0,Se,Be,ye.data)}else{O&&we&&t.texStorage2D(i.TEXTURE_2D,De,Ge,tt[0].width,tt[0].height);for(let de=0,le=tt.length;de<le;de++)ye=tt[de],_.format!==En?Se!==null?O?_e&&t.compressedTexSubImage2D(i.TEXTURE_2D,de,0,0,ye.width,ye.height,Se,ye.data):t.compressedTexImage2D(i.TEXTURE_2D,de,Ge,ye.width,ye.height,0,ye.data):et("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):O?_e&&t.texSubImage2D(i.TEXTURE_2D,de,0,0,ye.width,ye.height,Se,Be,ye.data):t.texImage2D(i.TEXTURE_2D,de,Ge,ye.width,ye.height,0,Se,Be,ye.data)}else if(_.isDataArrayTexture)if(O){if(we&&t.texStorage3D(i.TEXTURE_2D_ARRAY,De,Ge,ce.width,ce.height,ce.depth),_e)if(_.layerUpdates.size>0){const de=jo(ce.width,ce.height,_.format,_.type);for(const le of _.layerUpdates){const xe=ce.data.subarray(le*de/ce.data.BYTES_PER_ELEMENT,(le+1)*de/ce.data.BYTES_PER_ELEMENT);t.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,le,ce.width,ce.height,1,Se,Be,xe)}_.clearLayerUpdates()}else t.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,0,ce.width,ce.height,ce.depth,Se,Be,ce.data)}else t.texImage3D(i.TEXTURE_2D_ARRAY,0,Ge,ce.width,ce.height,ce.depth,0,Se,Be,ce.data);else if(_.isData3DTexture)O?(we&&t.texStorage3D(i.TEXTURE_3D,De,Ge,ce.width,ce.height,ce.depth),_e&&t.texSubImage3D(i.TEXTURE_3D,0,0,0,0,ce.width,ce.height,ce.depth,Se,Be,ce.data)):t.texImage3D(i.TEXTURE_3D,0,Ge,ce.width,ce.height,ce.depth,0,Se,Be,ce.data);else if(_.isFramebufferTexture){if(we)if(O)t.texStorage2D(i.TEXTURE_2D,De,Ge,ce.width,ce.height);else{let de=ce.width,le=ce.height;for(let xe=0;xe<De;xe++)t.texImage2D(i.TEXTURE_2D,xe,Ge,de,le,0,Se,Be,null),de>>=1,le>>=1}}else if(tt.length>0){if(O&&we){const de=Oe(tt[0]);t.texStorage2D(i.TEXTURE_2D,De,Ge,de.width,de.height)}for(let de=0,le=tt.length;de<le;de++)ye=tt[de],O?_e&&t.texSubImage2D(i.TEXTURE_2D,de,0,0,Se,Be,ye):t.texImage2D(i.TEXTURE_2D,de,Ge,Se,Be,ye);_.generateMipmaps=!1}else if(O){if(we){const de=Oe(ce);t.texStorage2D(i.TEXTURE_2D,De,Ge,de.width,de.height)}_e&&t.texSubImage2D(i.TEXTURE_2D,0,0,0,Se,Be,ce)}else t.texImage2D(i.TEXTURE_2D,0,Ge,Se,Be,ce);g(_)&&d(ne),Me.__version=ee.version,_.onUpdate&&_.onUpdate(_)}w.__version=_.version}function fe(w,_,V){if(_.image.length!==6)return;const ne=it(w,_),se=_.source;t.bindTexture(i.TEXTURE_CUBE_MAP,w.__webglTexture,i.TEXTURE0+V);const ee=n.get(se);if(se.version!==ee.__version||ne===!0){t.activeTexture(i.TEXTURE0+V);const Me=_t.getPrimaries(_t.workingColorSpace),ve=_.colorSpace===ri?null:_t.getPrimaries(_.colorSpace),me=_.colorSpace===ri||Me===ve?i.NONE:i.BROWSER_DEFAULT_WEBGL;i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,_.flipY),i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,_.premultiplyAlpha),i.pixelStorei(i.UNPACK_ALIGNMENT,_.unpackAlignment),i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,me);const Ye=_.isCompressedTexture||_.image[0].isCompressedTexture,ce=_.image[0]&&_.image[0].isDataTexture,Se=[];for(let le=0;le<6;le++)!Ye&&!ce?Se[le]=b(_.image[le],!0,r.maxCubemapSize):Se[le]=ce?_.image[le].image:_.image[le],Se[le]=gt(_,Se[le]);const Be=Se[0],Ge=s.convert(_.format,_.colorSpace),ye=s.convert(_.type),tt=C(_.internalFormat,Ge,ye,_.colorSpace),O=_.isVideoTexture!==!0,we=ee.__version===void 0||ne===!0,_e=se.dataReady;let De=P(_,Be);at(i.TEXTURE_CUBE_MAP,_);let de;if(Ye){O&&we&&t.texStorage2D(i.TEXTURE_CUBE_MAP,De,tt,Be.width,Be.height);for(let le=0;le<6;le++){de=Se[le].mipmaps;for(let xe=0;xe<de.length;xe++){const Ke=de[xe];_.format!==En?Ge!==null?O?_e&&t.compressedTexSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,xe,0,0,Ke.width,Ke.height,Ge,Ke.data):t.compressedTexImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,xe,tt,Ke.width,Ke.height,0,Ke.data):et("WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):O?_e&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,xe,0,0,Ke.width,Ke.height,Ge,ye,Ke.data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,xe,tt,Ke.width,Ke.height,0,Ge,ye,Ke.data)}}}else{if(de=_.mipmaps,O&&we){de.length>0&&De++;const le=Oe(Se[0]);t.texStorage2D(i.TEXTURE_CUBE_MAP,De,tt,le.width,le.height)}for(let le=0;le<6;le++)if(ce){O?_e&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,0,0,0,Se[le].width,Se[le].height,Ge,ye,Se[le].data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,0,tt,Se[le].width,Se[le].height,0,Ge,ye,Se[le].data);for(let xe=0;xe<de.length;xe++){const At=de[xe].image[le].image;O?_e&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,xe+1,0,0,At.width,At.height,Ge,ye,At.data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,xe+1,tt,At.width,At.height,0,Ge,ye,At.data)}}else{O?_e&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,0,0,0,Ge,ye,Se[le]):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,0,tt,Ge,ye,Se[le]);for(let xe=0;xe<de.length;xe++){const Ke=de[xe];O?_e&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,xe+1,0,0,Ge,ye,Ke.image[le]):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+le,xe+1,tt,Ge,ye,Ke.image[le])}}}g(_)&&d(i.TEXTURE_CUBE_MAP),ee.__version=se.version,_.onUpdate&&_.onUpdate(_)}w.__version=_.version}function Ne(w,_,V,ne,se,ee){const Me=s.convert(V.format,V.colorSpace),ve=s.convert(V.type),me=C(V.internalFormat,Me,ve,V.colorSpace),Ye=n.get(_),ce=n.get(V);if(ce.__renderTarget=_,!Ye.__hasExternalTextures){const Se=Math.max(1,_.width>>ee),Be=Math.max(1,_.height>>ee);se===i.TEXTURE_3D||se===i.TEXTURE_2D_ARRAY?t.texImage3D(se,ee,me,Se,Be,_.depth,0,Me,ve,null):t.texImage2D(se,ee,me,Se,Be,0,Me,ve,null)}t.bindFramebuffer(i.FRAMEBUFFER,w),It(_)?o.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,ne,se,ce.__webglTexture,0,F(_)):(se===i.TEXTURE_2D||se>=i.TEXTURE_CUBE_MAP_POSITIVE_X&&se<=i.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&i.framebufferTexture2D(i.FRAMEBUFFER,ne,se,ce.__webglTexture,ee),t.bindFramebuffer(i.FRAMEBUFFER,null)}function Qe(w,_,V){if(i.bindRenderbuffer(i.RENDERBUFFER,w),_.depthBuffer){const ne=_.depthTexture,se=ne&&ne.isDepthTexture?ne.type:null,ee=A(_.stencilBuffer,se),Me=_.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT;It(_)?o.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,F(_),ee,_.width,_.height):V?i.renderbufferStorageMultisample(i.RENDERBUFFER,F(_),ee,_.width,_.height):i.renderbufferStorage(i.RENDERBUFFER,ee,_.width,_.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,Me,i.RENDERBUFFER,w)}else{const ne=_.textures;for(let se=0;se<ne.length;se++){const ee=ne[se],Me=s.convert(ee.format,ee.colorSpace),ve=s.convert(ee.type),me=C(ee.internalFormat,Me,ve,ee.colorSpace);It(_)?o.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,F(_),me,_.width,_.height):V?i.renderbufferStorageMultisample(i.RENDERBUFFER,F(_),me,_.width,_.height):i.renderbufferStorage(i.RENDERBUFFER,me,_.width,_.height)}}i.bindRenderbuffer(i.RENDERBUFFER,null)}function Ve(w,_,V){const ne=_.isWebGLCubeRenderTarget===!0;if(t.bindFramebuffer(i.FRAMEBUFFER,w),!(_.depthTexture&&_.depthTexture.isDepthTexture))throw new Error("renderTarget.depthTexture must be an instance of THREE.DepthTexture");const se=n.get(_.depthTexture);if(se.__renderTarget=_,(!se.__webglTexture||_.depthTexture.image.width!==_.width||_.depthTexture.image.height!==_.height)&&(_.depthTexture.image.width=_.width,_.depthTexture.image.height=_.height,_.depthTexture.needsUpdate=!0),ne){if(se.__webglInit===void 0&&(se.__webglInit=!0,_.depthTexture.addEventListener("dispose",D)),se.__webglTexture===void 0){se.__webglTexture=i.createTexture(),t.bindTexture(i.TEXTURE_CUBE_MAP,se.__webglTexture),at(i.TEXTURE_CUBE_MAP,_.depthTexture);const Ye=s.convert(_.depthTexture.format),ce=s.convert(_.depthTexture.type);let Se;_.depthTexture.format===Kn?Se=i.DEPTH_COMPONENT24:_.depthTexture.format===Mi&&(Se=i.DEPTH24_STENCIL8);for(let Be=0;Be<6;Be++)i.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+Be,0,Se,_.width,_.height,0,Ye,ce,null)}}else oe(_.depthTexture,0);const ee=se.__webglTexture,Me=F(_),ve=ne?i.TEXTURE_CUBE_MAP_POSITIVE_X+V:i.TEXTURE_2D,me=_.depthTexture.format===Mi?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT;if(_.depthTexture.format===Kn)It(_)?o.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,me,ve,ee,0,Me):i.framebufferTexture2D(i.FRAMEBUFFER,me,ve,ee,0);else if(_.depthTexture.format===Mi)It(_)?o.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,me,ve,ee,0,Me):i.framebufferTexture2D(i.FRAMEBUFFER,me,ve,ee,0);else throw new Error("Unknown depthTexture format")}function ht(w){const _=n.get(w),V=w.isWebGLCubeRenderTarget===!0;if(_.__boundDepthTexture!==w.depthTexture){const ne=w.depthTexture;if(_.__depthDisposeCallback&&_.__depthDisposeCallback(),ne){const se=()=>{delete _.__boundDepthTexture,delete _.__depthDisposeCallback,ne.removeEventListener("dispose",se)};ne.addEventListener("dispose",se),_.__depthDisposeCallback=se}_.__boundDepthTexture=ne}if(w.depthTexture&&!_.__autoAllocateDepthBuffer)if(V)for(let ne=0;ne<6;ne++)Ve(_.__webglFramebuffer[ne],w,ne);else{const ne=w.texture.mipmaps;ne&&ne.length>0?Ve(_.__webglFramebuffer[0],w,0):Ve(_.__webglFramebuffer,w,0)}else if(V){_.__webglDepthbuffer=[];for(let ne=0;ne<6;ne++)if(t.bindFramebuffer(i.FRAMEBUFFER,_.__webglFramebuffer[ne]),_.__webglDepthbuffer[ne]===void 0)_.__webglDepthbuffer[ne]=i.createRenderbuffer(),Qe(_.__webglDepthbuffer[ne],w,!1);else{const se=w.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,ee=_.__webglDepthbuffer[ne];i.bindRenderbuffer(i.RENDERBUFFER,ee),i.framebufferRenderbuffer(i.FRAMEBUFFER,se,i.RENDERBUFFER,ee)}}else{const ne=w.texture.mipmaps;if(ne&&ne.length>0?t.bindFramebuffer(i.FRAMEBUFFER,_.__webglFramebuffer[0]):t.bindFramebuffer(i.FRAMEBUFFER,_.__webglFramebuffer),_.__webglDepthbuffer===void 0)_.__webglDepthbuffer=i.createRenderbuffer(),Qe(_.__webglDepthbuffer,w,!1);else{const se=w.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,ee=_.__webglDepthbuffer;i.bindRenderbuffer(i.RENDERBUFFER,ee),i.framebufferRenderbuffer(i.FRAMEBUFFER,se,i.RENDERBUFFER,ee)}}t.bindFramebuffer(i.FRAMEBUFFER,null)}function Vt(w,_,V){const ne=n.get(w);_!==void 0&&Ne(ne.__webglFramebuffer,w,w.texture,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,0),V!==void 0&&ht(w)}function Je(w){const _=w.texture,V=n.get(w),ne=n.get(_);w.addEventListener("dispose",I);const se=w.textures,ee=w.isWebGLCubeRenderTarget===!0,Me=se.length>1;if(Me||(ne.__webglTexture===void 0&&(ne.__webglTexture=i.createTexture()),ne.__version=_.version,a.memory.textures++),ee){V.__webglFramebuffer=[];for(let ve=0;ve<6;ve++)if(_.mipmaps&&_.mipmaps.length>0){V.__webglFramebuffer[ve]=[];for(let me=0;me<_.mipmaps.length;me++)V.__webglFramebuffer[ve][me]=i.createFramebuffer()}else V.__webglFramebuffer[ve]=i.createFramebuffer()}else{if(_.mipmaps&&_.mipmaps.length>0){V.__webglFramebuffer=[];for(let ve=0;ve<_.mipmaps.length;ve++)V.__webglFramebuffer[ve]=i.createFramebuffer()}else V.__webglFramebuffer=i.createFramebuffer();if(Me)for(let ve=0,me=se.length;ve<me;ve++){const Ye=n.get(se[ve]);Ye.__webglTexture===void 0&&(Ye.__webglTexture=i.createTexture(),a.memory.textures++)}if(w.samples>0&&It(w)===!1){V.__webglMultisampledFramebuffer=i.createFramebuffer(),V.__webglColorRenderbuffer=[],t.bindFramebuffer(i.FRAMEBUFFER,V.__webglMultisampledFramebuffer);for(let ve=0;ve<se.length;ve++){const me=se[ve];V.__webglColorRenderbuffer[ve]=i.createRenderbuffer(),i.bindRenderbuffer(i.RENDERBUFFER,V.__webglColorRenderbuffer[ve]);const Ye=s.convert(me.format,me.colorSpace),ce=s.convert(me.type),Se=C(me.internalFormat,Ye,ce,me.colorSpace,w.isXRRenderTarget===!0),Be=F(w);i.renderbufferStorageMultisample(i.RENDERBUFFER,Be,Se,w.width,w.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+ve,i.RENDERBUFFER,V.__webglColorRenderbuffer[ve])}i.bindRenderbuffer(i.RENDERBUFFER,null),w.depthBuffer&&(V.__webglDepthRenderbuffer=i.createRenderbuffer(),Qe(V.__webglDepthRenderbuffer,w,!0)),t.bindFramebuffer(i.FRAMEBUFFER,null)}}if(ee){t.bindTexture(i.TEXTURE_CUBE_MAP,ne.__webglTexture),at(i.TEXTURE_CUBE_MAP,_);for(let ve=0;ve<6;ve++)if(_.mipmaps&&_.mipmaps.length>0)for(let me=0;me<_.mipmaps.length;me++)Ne(V.__webglFramebuffer[ve][me],w,_,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+ve,me);else Ne(V.__webglFramebuffer[ve],w,_,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+ve,0);g(_)&&d(i.TEXTURE_CUBE_MAP),t.unbindTexture()}else if(Me){for(let ve=0,me=se.length;ve<me;ve++){const Ye=se[ve],ce=n.get(Ye);let Se=i.TEXTURE_2D;(w.isWebGL3DRenderTarget||w.isWebGLArrayRenderTarget)&&(Se=w.isWebGL3DRenderTarget?i.TEXTURE_3D:i.TEXTURE_2D_ARRAY),t.bindTexture(Se,ce.__webglTexture),at(Se,Ye),Ne(V.__webglFramebuffer,w,Ye,i.COLOR_ATTACHMENT0+ve,Se,0),g(Ye)&&d(Se)}t.unbindTexture()}else{let ve=i.TEXTURE_2D;if((w.isWebGL3DRenderTarget||w.isWebGLArrayRenderTarget)&&(ve=w.isWebGL3DRenderTarget?i.TEXTURE_3D:i.TEXTURE_2D_ARRAY),t.bindTexture(ve,ne.__webglTexture),at(ve,_),_.mipmaps&&_.mipmaps.length>0)for(let me=0;me<_.mipmaps.length;me++)Ne(V.__webglFramebuffer[me],w,_,i.COLOR_ATTACHMENT0,ve,me);else Ne(V.__webglFramebuffer,w,_,i.COLOR_ATTACHMENT0,ve,0);g(_)&&d(ve),t.unbindTexture()}w.depthBuffer&&ht(w)}function ut(w){const _=w.textures;for(let V=0,ne=_.length;V<ne;V++){const se=_[V];if(g(se)){const ee=R(w),Me=n.get(se).__webglTexture;t.bindTexture(ee,Me),d(ee),t.unbindTexture()}}}const Tt=[],nt=[];function Ft(w){if(w.samples>0){if(It(w)===!1){const _=w.textures,V=w.width,ne=w.height;let se=i.COLOR_BUFFER_BIT;const ee=w.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,Me=n.get(w),ve=_.length>1;if(ve)for(let Ye=0;Ye<_.length;Ye++)t.bindFramebuffer(i.FRAMEBUFFER,Me.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+Ye,i.RENDERBUFFER,null),t.bindFramebuffer(i.FRAMEBUFFER,Me.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+Ye,i.TEXTURE_2D,null,0);t.bindFramebuffer(i.READ_FRAMEBUFFER,Me.__webglMultisampledFramebuffer);const me=w.texture.mipmaps;me&&me.length>0?t.bindFramebuffer(i.DRAW_FRAMEBUFFER,Me.__webglFramebuffer[0]):t.bindFramebuffer(i.DRAW_FRAMEBUFFER,Me.__webglFramebuffer);for(let Ye=0;Ye<_.length;Ye++){if(w.resolveDepthBuffer&&(w.depthBuffer&&(se|=i.DEPTH_BUFFER_BIT),w.stencilBuffer&&w.resolveStencilBuffer&&(se|=i.STENCIL_BUFFER_BIT)),ve){i.framebufferRenderbuffer(i.READ_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.RENDERBUFFER,Me.__webglColorRenderbuffer[Ye]);const ce=n.get(_[Ye]).__webglTexture;i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,ce,0)}i.blitFramebuffer(0,0,V,ne,0,0,V,ne,se,i.NEAREST),u===!0&&(Tt.length=0,nt.length=0,Tt.push(i.COLOR_ATTACHMENT0+Ye),w.depthBuffer&&w.resolveDepthBuffer===!1&&(Tt.push(ee),nt.push(ee),i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER,nt)),i.invalidateFramebuffer(i.READ_FRAMEBUFFER,Tt))}if(t.bindFramebuffer(i.READ_FRAMEBUFFER,null),t.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),ve)for(let Ye=0;Ye<_.length;Ye++){t.bindFramebuffer(i.FRAMEBUFFER,Me.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+Ye,i.RENDERBUFFER,Me.__webglColorRenderbuffer[Ye]);const ce=n.get(_[Ye]).__webglTexture;t.bindFramebuffer(i.FRAMEBUFFER,Me.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+Ye,i.TEXTURE_2D,ce,0)}t.bindFramebuffer(i.DRAW_FRAMEBUFFER,Me.__webglMultisampledFramebuffer)}else if(w.depthBuffer&&w.resolveDepthBuffer===!1&&u){const _=w.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT;i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER,[_])}}}function F(w){return Math.min(r.maxSamples,w.samples)}function It(w){const _=n.get(w);return w.samples>0&&e.has("WEBGL_multisampled_render_to_texture")===!0&&_.__useRenderToTexture!==!1}function mt(w){const _=a.render.frame;h.get(w)!==_&&(h.set(w,_),w.update())}function gt(w,_){const V=w.colorSpace,ne=w.format,se=w.type;return w.isCompressedTexture===!0||w.isVideoTexture===!0||V!==Xi&&V!==ri&&(_t.getTransfer(V)===wt?(ne!==En||se!==dn)&&et("WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):xt("WebGLTextures: Unsupported texture color space:",V)),_}function Oe(w){return typeof HTMLImageElement<"u"&&w instanceof HTMLImageElement?(l.width=w.naturalWidth||w.width,l.height=w.naturalHeight||w.height):typeof VideoFrame<"u"&&w instanceof VideoFrame?(l.width=w.displayWidth,l.height=w.displayHeight):(l.width=w.width,l.height=w.height),l}this.allocateTextureUnit=K,this.resetTextureUnits=j,this.setTexture2D=oe,this.setTexture2DArray=Q,this.setTexture3D=te,this.setTextureCube=ue,this.rebindTextures=Vt,this.setupRenderTarget=Je,this.updateRenderTargetMipmap=ut,this.updateMultisampleRenderTarget=Ft,this.setupDepthRenderbuffer=ht,this.setupFrameBufferTexture=Ne,this.useMultisampledRTT=It,this.isReversedDepthBuffer=function(){return t.buffers.depth.getReversed()}}function u_(i,e){function t(n,r=ri){let s;const a=_t.getTransfer(r);if(n===dn)return i.UNSIGNED_BYTE;if(n===ja)return i.UNSIGNED_SHORT_4_4_4_4;if(n===Za)return i.UNSIGNED_SHORT_5_5_5_1;if(n===Ll)return i.UNSIGNED_INT_5_9_9_9_REV;if(n===Ul)return i.UNSIGNED_INT_10F_11F_11F_REV;if(n===Pl)return i.BYTE;if(n===Dl)return i.SHORT;if(n===hr)return i.UNSIGNED_SHORT;if(n===Ka)return i.INT;if(n===Ln)return i.UNSIGNED_INT;if(n===Rn)return i.FLOAT;if(n===Yn)return i.HALF_FLOAT;if(n===Il)return i.ALPHA;if(n===Fl)return i.RGB;if(n===En)return i.RGBA;if(n===Kn)return i.DEPTH_COMPONENT;if(n===Mi)return i.DEPTH_STENCIL;if(n===Nl)return i.RED;if(n===Ja)return i.RED_INTEGER;if(n===Wi)return i.RG;if(n===Qa)return i.RG_INTEGER;if(n===eo)return i.RGBA_INTEGER;if(n===Zr||n===Jr||n===Qr||n===es)if(a===wt)if(s=e.get("WEBGL_compressed_texture_s3tc_srgb"),s!==null){if(n===Zr)return s.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(n===Jr)return s.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(n===Qr)return s.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(n===es)return s.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(s=e.get("WEBGL_compressed_texture_s3tc"),s!==null){if(n===Zr)return s.COMPRESSED_RGB_S3TC_DXT1_EXT;if(n===Jr)return s.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(n===Qr)return s.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(n===es)return s.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(n===fa||n===ha||n===da||n===pa)if(s=e.get("WEBGL_compressed_texture_pvrtc"),s!==null){if(n===fa)return s.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(n===ha)return s.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(n===da)return s.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(n===pa)return s.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(n===ma||n===_a||n===ga||n===va||n===xa||n===Ma||n===Sa)if(s=e.get("WEBGL_compressed_texture_etc"),s!==null){if(n===ma||n===_a)return a===wt?s.COMPRESSED_SRGB8_ETC2:s.COMPRESSED_RGB8_ETC2;if(n===ga)return a===wt?s.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:s.COMPRESSED_RGBA8_ETC2_EAC;if(n===va)return s.COMPRESSED_R11_EAC;if(n===xa)return s.COMPRESSED_SIGNED_R11_EAC;if(n===Ma)return s.COMPRESSED_RG11_EAC;if(n===Sa)return s.COMPRESSED_SIGNED_RG11_EAC}else return null;if(n===ya||n===Ea||n===ba||n===Ta||n===Aa||n===wa||n===Ra||n===Ca||n===Pa||n===Da||n===La||n===Ua||n===Ia||n===Fa)if(s=e.get("WEBGL_compressed_texture_astc"),s!==null){if(n===ya)return a===wt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:s.COMPRESSED_RGBA_ASTC_4x4_KHR;if(n===Ea)return a===wt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:s.COMPRESSED_RGBA_ASTC_5x4_KHR;if(n===ba)return a===wt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:s.COMPRESSED_RGBA_ASTC_5x5_KHR;if(n===Ta)return a===wt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:s.COMPRESSED_RGBA_ASTC_6x5_KHR;if(n===Aa)return a===wt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:s.COMPRESSED_RGBA_ASTC_6x6_KHR;if(n===wa)return a===wt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:s.COMPRESSED_RGBA_ASTC_8x5_KHR;if(n===Ra)return a===wt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:s.COMPRESSED_RGBA_ASTC_8x6_KHR;if(n===Ca)return a===wt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:s.COMPRESSED_RGBA_ASTC_8x8_KHR;if(n===Pa)return a===wt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:s.COMPRESSED_RGBA_ASTC_10x5_KHR;if(n===Da)return a===wt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:s.COMPRESSED_RGBA_ASTC_10x6_KHR;if(n===La)return a===wt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:s.COMPRESSED_RGBA_ASTC_10x8_KHR;if(n===Ua)return a===wt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:s.COMPRESSED_RGBA_ASTC_10x10_KHR;if(n===Ia)return a===wt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:s.COMPRESSED_RGBA_ASTC_12x10_KHR;if(n===Fa)return a===wt?s.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:s.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(n===Na||n===Oa||n===Ba)if(s=e.get("EXT_texture_compression_bptc"),s!==null){if(n===Na)return a===wt?s.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:s.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(n===Oa)return s.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(n===Ba)return s.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(n===Va||n===za||n===Ga||n===Ha)if(s=e.get("EXT_texture_compression_rgtc"),s!==null){if(n===Va)return s.COMPRESSED_RED_RGTC1_EXT;if(n===za)return s.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(n===Ga)return s.COMPRESSED_RED_GREEN_RGTC2_EXT;if(n===Ha)return s.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return n===dr?i.UNSIGNED_INT_24_8:i[n]!==void 0?i[n]:null}return{convert:t}}const f_=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,h_=`
uniform sampler2DArray depthColor;
uniform float depthWidth;
uniform float depthHeight;

void main() {

	vec2 coord = vec2( gl_FragCoord.x / depthWidth, gl_FragCoord.y / depthHeight );

	if ( coord.x >= 1.0 ) {

		gl_FragDepth = texture( depthColor, vec3( coord.x - 1.0, coord.y, 1 ) ).r;

	} else {

		gl_FragDepth = texture( depthColor, vec3( coord.x, coord.y, 0 ) ).r;

	}

}`;class d_{constructor(){this.texture=null,this.mesh=null,this.depthNear=0,this.depthFar=0}init(e,t){if(this.texture===null){const n=new Yl(e.texture);(e.depthNear!==t.depthNear||e.depthFar!==t.depthFar)&&(this.depthNear=e.depthNear,this.depthFar=e.depthFar),this.texture=n}}getMesh(e){if(this.texture!==null&&this.mesh===null){const t=e.cameras[0].viewport,n=new Fn({vertexShader:f_,fragmentShader:h_,uniforms:{depthColor:{value:this.texture},depthWidth:{value:t.z},depthHeight:{value:t.w}}});this.mesh=new In(new as(20,20),n)}return this.mesh}reset(){this.texture=null,this.mesh=null}getDepthTexture(){return this.texture}}class p_ extends qi{constructor(e,t){super();const n=this;let r=null,s=1,a=null,o="local-floor",u=1,l=null,h=null,p=null,m=null,v=null,y=null;const b=typeof XRWebGLBinding<"u",g=new d_,d={},R=t.getContextAttributes();let C=null,A=null;const P=[],D=[],I=new pt;let G=null;const x=new hn;x.viewport=new Nt;const E=new hn;E.viewport=new Nt;const B=[x,E],j=new bf;let K=null,ie=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(re){let fe=P[re];return fe===void 0&&(fe=new ks,P[re]=fe),fe.getTargetRaySpace()},this.getControllerGrip=function(re){let fe=P[re];return fe===void 0&&(fe=new ks,P[re]=fe),fe.getGripSpace()},this.getHand=function(re){let fe=P[re];return fe===void 0&&(fe=new ks,P[re]=fe),fe.getHandSpace()};function oe(re){const fe=D.indexOf(re.inputSource);if(fe===-1)return;const Ne=P[fe];Ne!==void 0&&(Ne.update(re.inputSource,re.frame,l||a),Ne.dispatchEvent({type:re.type,data:re.inputSource}))}function Q(){r.removeEventListener("select",oe),r.removeEventListener("selectstart",oe),r.removeEventListener("selectend",oe),r.removeEventListener("squeeze",oe),r.removeEventListener("squeezestart",oe),r.removeEventListener("squeezeend",oe),r.removeEventListener("end",Q),r.removeEventListener("inputsourceschange",te);for(let re=0;re<P.length;re++){const fe=D[re];fe!==null&&(D[re]=null,P[re].disconnect(fe))}K=null,ie=null,g.reset();for(const re in d)delete d[re];e.setRenderTarget(C),v=null,m=null,p=null,r=null,A=null,Dt.stop(),n.isPresenting=!1,e.setPixelRatio(G),e.setSize(I.width,I.height,!1),n.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function(re){s=re,n.isPresenting===!0&&et("WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(re){o=re,n.isPresenting===!0&&et("WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return l||a},this.setReferenceSpace=function(re){l=re},this.getBaseLayer=function(){return m!==null?m:v},this.getBinding=function(){return p===null&&b&&(p=new XRWebGLBinding(r,t)),p},this.getFrame=function(){return y},this.getSession=function(){return r},this.setSession=async function(re){if(r=re,r!==null){if(C=e.getRenderTarget(),r.addEventListener("select",oe),r.addEventListener("selectstart",oe),r.addEventListener("selectend",oe),r.addEventListener("squeeze",oe),r.addEventListener("squeezestart",oe),r.addEventListener("squeezeend",oe),r.addEventListener("end",Q),r.addEventListener("inputsourceschange",te),R.xrCompatible!==!0&&await t.makeXRCompatible(),G=e.getPixelRatio(),e.getSize(I),b&&"createProjectionLayer"in XRWebGLBinding.prototype){let Ne=null,Qe=null,Ve=null;R.depth&&(Ve=R.stencil?t.DEPTH24_STENCIL8:t.DEPTH_COMPONENT24,Ne=R.stencil?Mi:Kn,Qe=R.stencil?dr:Ln);const ht={colorFormat:t.RGBA8,depthFormat:Ve,scaleFactor:s};p=this.getBinding(),m=p.createProjectionLayer(ht),r.updateRenderState({layers:[m]}),e.setPixelRatio(1),e.setSize(m.textureWidth,m.textureHeight,!1),A=new Dn(m.textureWidth,m.textureHeight,{format:En,type:dn,depthTexture:new mr(m.textureWidth,m.textureHeight,Qe,void 0,void 0,void 0,void 0,void 0,void 0,Ne),stencilBuffer:R.stencil,colorSpace:e.outputColorSpace,samples:R.antialias?4:0,resolveDepthBuffer:m.ignoreDepthValues===!1,resolveStencilBuffer:m.ignoreDepthValues===!1})}else{const Ne={antialias:R.antialias,alpha:!0,depth:R.depth,stencil:R.stencil,framebufferScaleFactor:s};v=new XRWebGLLayer(r,t,Ne),r.updateRenderState({baseLayer:v}),e.setPixelRatio(1),e.setSize(v.framebufferWidth,v.framebufferHeight,!1),A=new Dn(v.framebufferWidth,v.framebufferHeight,{format:En,type:dn,colorSpace:e.outputColorSpace,stencilBuffer:R.stencil,resolveDepthBuffer:v.ignoreDepthValues===!1,resolveStencilBuffer:v.ignoreDepthValues===!1})}A.isXRRenderTarget=!0,this.setFoveation(u),l=null,a=await r.requestReferenceSpace(o),Dt.setContext(r),Dt.start(),n.isPresenting=!0,n.dispatchEvent({type:"sessionstart"})}},this.getEnvironmentBlendMode=function(){if(r!==null)return r.environmentBlendMode},this.getDepthTexture=function(){return g.getDepthTexture()};function te(re){for(let fe=0;fe<re.removed.length;fe++){const Ne=re.removed[fe],Qe=D.indexOf(Ne);Qe>=0&&(D[Qe]=null,P[Qe].disconnect(Ne))}for(let fe=0;fe<re.added.length;fe++){const Ne=re.added[fe];let Qe=D.indexOf(Ne);if(Qe===-1){for(let ht=0;ht<P.length;ht++)if(ht>=D.length){D.push(Ne),Qe=ht;break}else if(D[ht]===null){D[ht]=Ne,Qe=ht;break}if(Qe===-1)break}const Ve=P[Qe];Ve&&Ve.connect(Ne)}}const ue=new $,Ae=new $;function Ee(re,fe,Ne){ue.setFromMatrixPosition(fe.matrixWorld),Ae.setFromMatrixPosition(Ne.matrixWorld);const Qe=ue.distanceTo(Ae),Ve=fe.projectionMatrix.elements,ht=Ne.projectionMatrix.elements,Vt=Ve[14]/(Ve[10]-1),Je=Ve[14]/(Ve[10]+1),ut=(Ve[9]+1)/Ve[5],Tt=(Ve[9]-1)/Ve[5],nt=(Ve[8]-1)/Ve[0],Ft=(ht[8]+1)/ht[0],F=Vt*nt,It=Vt*Ft,mt=Qe/(-nt+Ft),gt=mt*-nt;if(fe.matrixWorld.decompose(re.position,re.quaternion,re.scale),re.translateX(gt),re.translateZ(mt),re.matrixWorld.compose(re.position,re.quaternion,re.scale),re.matrixWorldInverse.copy(re.matrixWorld).invert(),Ve[10]===-1)re.projectionMatrix.copy(fe.projectionMatrix),re.projectionMatrixInverse.copy(fe.projectionMatrixInverse);else{const Oe=Vt+mt,w=Je+mt,_=F-gt,V=It+(Qe-gt),ne=ut*Je/w*Oe,se=Tt*Je/w*Oe;re.projectionMatrix.makePerspective(_,V,ne,se,Oe,w),re.projectionMatrixInverse.copy(re.projectionMatrix).invert()}}function Le(re,fe){fe===null?re.matrixWorld.copy(re.matrix):re.matrixWorld.multiplyMatrices(fe.matrixWorld,re.matrix),re.matrixWorldInverse.copy(re.matrixWorld).invert()}this.updateCamera=function(re){if(r===null)return;let fe=re.near,Ne=re.far;g.texture!==null&&(g.depthNear>0&&(fe=g.depthNear),g.depthFar>0&&(Ne=g.depthFar)),j.near=E.near=x.near=fe,j.far=E.far=x.far=Ne,(K!==j.near||ie!==j.far)&&(r.updateRenderState({depthNear:j.near,depthFar:j.far}),K=j.near,ie=j.far),j.layers.mask=re.layers.mask|6,x.layers.mask=j.layers.mask&3,E.layers.mask=j.layers.mask&5;const Qe=re.parent,Ve=j.cameras;Le(j,Qe);for(let ht=0;ht<Ve.length;ht++)Le(Ve[ht],Qe);Ve.length===2?Ee(j,x,E):j.projectionMatrix.copy(x.projectionMatrix),at(re,j,Qe)};function at(re,fe,Ne){Ne===null?re.matrix.copy(fe.matrixWorld):(re.matrix.copy(Ne.matrixWorld),re.matrix.invert(),re.matrix.multiply(fe.matrixWorld)),re.matrix.decompose(re.position,re.quaternion,re.scale),re.updateMatrixWorld(!0),re.projectionMatrix.copy(fe.projectionMatrix),re.projectionMatrixInverse.copy(fe.projectionMatrixInverse),re.isPerspectiveCamera&&(re.fov=ka*2*Math.atan(1/re.projectionMatrix.elements[5]),re.zoom=1)}this.getCamera=function(){return j},this.getFoveation=function(){if(!(m===null&&v===null))return u},this.setFoveation=function(re){u=re,m!==null&&(m.fixedFoveation=re),v!==null&&v.fixedFoveation!==void 0&&(v.fixedFoveation=re)},this.hasDepthSensing=function(){return g.texture!==null},this.getDepthSensingMesh=function(){return g.getMesh(j)},this.getCameraTexture=function(re){return d[re]};let it=null;function Ut(re,fe){if(h=fe.getViewerPose(l||a),y=fe,h!==null){const Ne=h.views;v!==null&&(e.setRenderTargetFramebuffer(A,v.framebuffer),e.setRenderTarget(A));let Qe=!1;Ne.length!==j.cameras.length&&(j.cameras.length=0,Qe=!0);for(let Je=0;Je<Ne.length;Je++){const ut=Ne[Je];let Tt=null;if(v!==null)Tt=v.getViewport(ut);else{const Ft=p.getViewSubImage(m,ut);Tt=Ft.viewport,Je===0&&(e.setRenderTargetTextures(A,Ft.colorTexture,Ft.depthStencilTexture),e.setRenderTarget(A))}let nt=B[Je];nt===void 0&&(nt=new hn,nt.layers.enable(Je),nt.viewport=new Nt,B[Je]=nt),nt.matrix.fromArray(ut.transform.matrix),nt.matrix.decompose(nt.position,nt.quaternion,nt.scale),nt.projectionMatrix.fromArray(ut.projectionMatrix),nt.projectionMatrixInverse.copy(nt.projectionMatrix).invert(),nt.viewport.set(Tt.x,Tt.y,Tt.width,Tt.height),Je===0&&(j.matrix.copy(nt.matrix),j.matrix.decompose(j.position,j.quaternion,j.scale)),Qe===!0&&j.cameras.push(nt)}const Ve=r.enabledFeatures;if(Ve&&Ve.includes("depth-sensing")&&r.depthUsage=="gpu-optimized"&&b){p=n.getBinding();const Je=p.getDepthInformation(Ne[0]);Je&&Je.isValid&&Je.texture&&g.init(Je,r.renderState)}if(Ve&&Ve.includes("camera-access")&&b){e.state.unbindTexture(),p=n.getBinding();for(let Je=0;Je<Ne.length;Je++){const ut=Ne[Je].camera;if(ut){let Tt=d[ut];Tt||(Tt=new Yl,d[ut]=Tt);const nt=p.getCameraImage(ut);Tt.sourceTexture=nt}}}}for(let Ne=0;Ne<P.length;Ne++){const Qe=D[Ne],Ve=P[Ne];Qe!==null&&Ve!==void 0&&Ve.update(Qe,fe,l||a)}it&&it(re,fe),fe.detectedPlanes&&n.dispatchEvent({type:"planesdetected",data:fe}),y=null}const Dt=new jl;Dt.setAnimationLoop(Ut),this.setAnimationLoop=function(re){it=re},this.dispose=function(){}}}const mi=new Un,m_=new Ot;function __(i,e){function t(g,d){g.matrixAutoUpdate===!0&&g.updateMatrix(),d.value.copy(g.matrix)}function n(g,d){d.color.getRGB(g.fogColor.value,Wl(i)),d.isFog?(g.fogNear.value=d.near,g.fogFar.value=d.far):d.isFogExp2&&(g.fogDensity.value=d.density)}function r(g,d,R,C,A){d.isMeshBasicMaterial||d.isMeshLambertMaterial?s(g,d):d.isMeshToonMaterial?(s(g,d),p(g,d)):d.isMeshPhongMaterial?(s(g,d),h(g,d)):d.isMeshStandardMaterial?(s(g,d),m(g,d),d.isMeshPhysicalMaterial&&v(g,d,A)):d.isMeshMatcapMaterial?(s(g,d),y(g,d)):d.isMeshDepthMaterial?s(g,d):d.isMeshDistanceMaterial?(s(g,d),b(g,d)):d.isMeshNormalMaterial?s(g,d):d.isLineBasicMaterial?(a(g,d),d.isLineDashedMaterial&&o(g,d)):d.isPointsMaterial?u(g,d,R,C):d.isSpriteMaterial?l(g,d):d.isShadowMaterial?(g.color.value.copy(d.color),g.opacity.value=d.opacity):d.isShaderMaterial&&(d.uniformsNeedUpdate=!1)}function s(g,d){g.opacity.value=d.opacity,d.color&&g.diffuse.value.copy(d.color),d.emissive&&g.emissive.value.copy(d.emissive).multiplyScalar(d.emissiveIntensity),d.map&&(g.map.value=d.map,t(d.map,g.mapTransform)),d.alphaMap&&(g.alphaMap.value=d.alphaMap,t(d.alphaMap,g.alphaMapTransform)),d.bumpMap&&(g.bumpMap.value=d.bumpMap,t(d.bumpMap,g.bumpMapTransform),g.bumpScale.value=d.bumpScale,d.side===on&&(g.bumpScale.value*=-1)),d.normalMap&&(g.normalMap.value=d.normalMap,t(d.normalMap,g.normalMapTransform),g.normalScale.value.copy(d.normalScale),d.side===on&&g.normalScale.value.negate()),d.displacementMap&&(g.displacementMap.value=d.displacementMap,t(d.displacementMap,g.displacementMapTransform),g.displacementScale.value=d.displacementScale,g.displacementBias.value=d.displacementBias),d.emissiveMap&&(g.emissiveMap.value=d.emissiveMap,t(d.emissiveMap,g.emissiveMapTransform)),d.specularMap&&(g.specularMap.value=d.specularMap,t(d.specularMap,g.specularMapTransform)),d.alphaTest>0&&(g.alphaTest.value=d.alphaTest);const R=e.get(d),C=R.envMap,A=R.envMapRotation;C&&(g.envMap.value=C,mi.copy(A),mi.x*=-1,mi.y*=-1,mi.z*=-1,C.isCubeTexture&&C.isRenderTargetTexture===!1&&(mi.y*=-1,mi.z*=-1),g.envMapRotation.value.setFromMatrix4(m_.makeRotationFromEuler(mi)),g.flipEnvMap.value=C.isCubeTexture&&C.isRenderTargetTexture===!1?-1:1,g.reflectivity.value=d.reflectivity,g.ior.value=d.ior,g.refractionRatio.value=d.refractionRatio),d.lightMap&&(g.lightMap.value=d.lightMap,g.lightMapIntensity.value=d.lightMapIntensity,t(d.lightMap,g.lightMapTransform)),d.aoMap&&(g.aoMap.value=d.aoMap,g.aoMapIntensity.value=d.aoMapIntensity,t(d.aoMap,g.aoMapTransform))}function a(g,d){g.diffuse.value.copy(d.color),g.opacity.value=d.opacity,d.map&&(g.map.value=d.map,t(d.map,g.mapTransform))}function o(g,d){g.dashSize.value=d.dashSize,g.totalSize.value=d.dashSize+d.gapSize,g.scale.value=d.scale}function u(g,d,R,C){g.diffuse.value.copy(d.color),g.opacity.value=d.opacity,g.size.value=d.size*R,g.scale.value=C*.5,d.map&&(g.map.value=d.map,t(d.map,g.uvTransform)),d.alphaMap&&(g.alphaMap.value=d.alphaMap,t(d.alphaMap,g.alphaMapTransform)),d.alphaTest>0&&(g.alphaTest.value=d.alphaTest)}function l(g,d){g.diffuse.value.copy(d.color),g.opacity.value=d.opacity,g.rotation.value=d.rotation,d.map&&(g.map.value=d.map,t(d.map,g.mapTransform)),d.alphaMap&&(g.alphaMap.value=d.alphaMap,t(d.alphaMap,g.alphaMapTransform)),d.alphaTest>0&&(g.alphaTest.value=d.alphaTest)}function h(g,d){g.specular.value.copy(d.specular),g.shininess.value=Math.max(d.shininess,1e-4)}function p(g,d){d.gradientMap&&(g.gradientMap.value=d.gradientMap)}function m(g,d){g.metalness.value=d.metalness,d.metalnessMap&&(g.metalnessMap.value=d.metalnessMap,t(d.metalnessMap,g.metalnessMapTransform)),g.roughness.value=d.roughness,d.roughnessMap&&(g.roughnessMap.value=d.roughnessMap,t(d.roughnessMap,g.roughnessMapTransform)),d.envMap&&(g.envMapIntensity.value=d.envMapIntensity)}function v(g,d,R){g.ior.value=d.ior,d.sheen>0&&(g.sheenColor.value.copy(d.sheenColor).multiplyScalar(d.sheen),g.sheenRoughness.value=d.sheenRoughness,d.sheenColorMap&&(g.sheenColorMap.value=d.sheenColorMap,t(d.sheenColorMap,g.sheenColorMapTransform)),d.sheenRoughnessMap&&(g.sheenRoughnessMap.value=d.sheenRoughnessMap,t(d.sheenRoughnessMap,g.sheenRoughnessMapTransform))),d.clearcoat>0&&(g.clearcoat.value=d.clearcoat,g.clearcoatRoughness.value=d.clearcoatRoughness,d.clearcoatMap&&(g.clearcoatMap.value=d.clearcoatMap,t(d.clearcoatMap,g.clearcoatMapTransform)),d.clearcoatRoughnessMap&&(g.clearcoatRoughnessMap.value=d.clearcoatRoughnessMap,t(d.clearcoatRoughnessMap,g.clearcoatRoughnessMapTransform)),d.clearcoatNormalMap&&(g.clearcoatNormalMap.value=d.clearcoatNormalMap,t(d.clearcoatNormalMap,g.clearcoatNormalMapTransform),g.clearcoatNormalScale.value.copy(d.clearcoatNormalScale),d.side===on&&g.clearcoatNormalScale.value.negate())),d.dispersion>0&&(g.dispersion.value=d.dispersion),d.iridescence>0&&(g.iridescence.value=d.iridescence,g.iridescenceIOR.value=d.iridescenceIOR,g.iridescenceThicknessMinimum.value=d.iridescenceThicknessRange[0],g.iridescenceThicknessMaximum.value=d.iridescenceThicknessRange[1],d.iridescenceMap&&(g.iridescenceMap.value=d.iridescenceMap,t(d.iridescenceMap,g.iridescenceMapTransform)),d.iridescenceThicknessMap&&(g.iridescenceThicknessMap.value=d.iridescenceThicknessMap,t(d.iridescenceThicknessMap,g.iridescenceThicknessMapTransform))),d.transmission>0&&(g.transmission.value=d.transmission,g.transmissionSamplerMap.value=R.texture,g.transmissionSamplerSize.value.set(R.width,R.height),d.transmissionMap&&(g.transmissionMap.value=d.transmissionMap,t(d.transmissionMap,g.transmissionMapTransform)),g.thickness.value=d.thickness,d.thicknessMap&&(g.thicknessMap.value=d.thicknessMap,t(d.thicknessMap,g.thicknessMapTransform)),g.attenuationDistance.value=d.attenuationDistance,g.attenuationColor.value.copy(d.attenuationColor)),d.anisotropy>0&&(g.anisotropyVector.value.set(d.anisotropy*Math.cos(d.anisotropyRotation),d.anisotropy*Math.sin(d.anisotropyRotation)),d.anisotropyMap&&(g.anisotropyMap.value=d.anisotropyMap,t(d.anisotropyMap,g.anisotropyMapTransform))),g.specularIntensity.value=d.specularIntensity,g.specularColor.value.copy(d.specularColor),d.specularColorMap&&(g.specularColorMap.value=d.specularColorMap,t(d.specularColorMap,g.specularColorMapTransform)),d.specularIntensityMap&&(g.specularIntensityMap.value=d.specularIntensityMap,t(d.specularIntensityMap,g.specularIntensityMapTransform))}function y(g,d){d.matcap&&(g.matcap.value=d.matcap)}function b(g,d){const R=e.get(d).light;g.referencePosition.value.setFromMatrixPosition(R.matrixWorld),g.nearDistance.value=R.shadow.camera.near,g.farDistance.value=R.shadow.camera.far}return{refreshFogUniforms:n,refreshMaterialUniforms:r}}function g_(i,e,t,n){let r={},s={},a=[];const o=i.getParameter(i.MAX_UNIFORM_BUFFER_BINDINGS);function u(R,C){const A=C.program;n.uniformBlockBinding(R,A)}function l(R,C){let A=r[R.id];A===void 0&&(y(R),A=h(R),r[R.id]=A,R.addEventListener("dispose",g));const P=C.program;n.updateUBOMapping(R,P);const D=e.render.frame;s[R.id]!==D&&(m(R),s[R.id]=D)}function h(R){const C=p();R.__bindingPointIndex=C;const A=i.createBuffer(),P=R.__size,D=R.usage;return i.bindBuffer(i.UNIFORM_BUFFER,A),i.bufferData(i.UNIFORM_BUFFER,P,D),i.bindBuffer(i.UNIFORM_BUFFER,null),i.bindBufferBase(i.UNIFORM_BUFFER,C,A),A}function p(){for(let R=0;R<o;R++)if(a.indexOf(R)===-1)return a.push(R),R;return xt("WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."),0}function m(R){const C=r[R.id],A=R.uniforms,P=R.__cache;i.bindBuffer(i.UNIFORM_BUFFER,C);for(let D=0,I=A.length;D<I;D++){const G=Array.isArray(A[D])?A[D]:[A[D]];for(let x=0,E=G.length;x<E;x++){const B=G[x];if(v(B,D,x,P)===!0){const j=B.__offset,K=Array.isArray(B.value)?B.value:[B.value];let ie=0;for(let oe=0;oe<K.length;oe++){const Q=K[oe],te=b(Q);typeof Q=="number"||typeof Q=="boolean"?(B.__data[0]=Q,i.bufferSubData(i.UNIFORM_BUFFER,j+ie,B.__data)):Q.isMatrix3?(B.__data[0]=Q.elements[0],B.__data[1]=Q.elements[1],B.__data[2]=Q.elements[2],B.__data[3]=0,B.__data[4]=Q.elements[3],B.__data[5]=Q.elements[4],B.__data[6]=Q.elements[5],B.__data[7]=0,B.__data[8]=Q.elements[6],B.__data[9]=Q.elements[7],B.__data[10]=Q.elements[8],B.__data[11]=0):(Q.toArray(B.__data,ie),ie+=te.storage/Float32Array.BYTES_PER_ELEMENT)}i.bufferSubData(i.UNIFORM_BUFFER,j,B.__data)}}}i.bindBuffer(i.UNIFORM_BUFFER,null)}function v(R,C,A,P){const D=R.value,I=C+"_"+A;if(P[I]===void 0)return typeof D=="number"||typeof D=="boolean"?P[I]=D:P[I]=D.clone(),!0;{const G=P[I];if(typeof D=="number"||typeof D=="boolean"){if(G!==D)return P[I]=D,!0}else if(G.equals(D)===!1)return G.copy(D),!0}return!1}function y(R){const C=R.uniforms;let A=0;const P=16;for(let I=0,G=C.length;I<G;I++){const x=Array.isArray(C[I])?C[I]:[C[I]];for(let E=0,B=x.length;E<B;E++){const j=x[E],K=Array.isArray(j.value)?j.value:[j.value];for(let ie=0,oe=K.length;ie<oe;ie++){const Q=K[ie],te=b(Q),ue=A%P,Ae=ue%te.boundary,Ee=ue+Ae;A+=Ae,Ee!==0&&P-Ee<te.storage&&(A+=P-Ee),j.__data=new Float32Array(te.storage/Float32Array.BYTES_PER_ELEMENT),j.__offset=A,A+=te.storage}}}const D=A%P;return D>0&&(A+=P-D),R.__size=A,R.__cache={},this}function b(R){const C={boundary:0,storage:0};return typeof R=="number"||typeof R=="boolean"?(C.boundary=4,C.storage=4):R.isVector2?(C.boundary=8,C.storage=8):R.isVector3||R.isColor?(C.boundary=16,C.storage=12):R.isVector4?(C.boundary=16,C.storage=16):R.isMatrix3?(C.boundary=48,C.storage=48):R.isMatrix4?(C.boundary=64,C.storage=64):R.isTexture?et("WebGLRenderer: Texture samplers can not be part of an uniforms group."):et("WebGLRenderer: Unsupported uniform value type.",R),C}function g(R){const C=R.target;C.removeEventListener("dispose",g);const A=a.indexOf(C.__bindingPointIndex);a.splice(A,1),i.deleteBuffer(r[C.id]),delete r[C.id],delete s[C.id]}function d(){for(const R in r)i.deleteBuffer(r[R]);a=[],r={},s={}}return{bind:u,update:l,dispose:d}}const v_=new Uint16Array([12469,15057,12620,14925,13266,14620,13807,14376,14323,13990,14545,13625,14713,13328,14840,12882,14931,12528,14996,12233,15039,11829,15066,11525,15080,11295,15085,10976,15082,10705,15073,10495,13880,14564,13898,14542,13977,14430,14158,14124,14393,13732,14556,13410,14702,12996,14814,12596,14891,12291,14937,11834,14957,11489,14958,11194,14943,10803,14921,10506,14893,10278,14858,9960,14484,14039,14487,14025,14499,13941,14524,13740,14574,13468,14654,13106,14743,12678,14818,12344,14867,11893,14889,11509,14893,11180,14881,10751,14852,10428,14812,10128,14765,9754,14712,9466,14764,13480,14764,13475,14766,13440,14766,13347,14769,13070,14786,12713,14816,12387,14844,11957,14860,11549,14868,11215,14855,10751,14825,10403,14782,10044,14729,9651,14666,9352,14599,9029,14967,12835,14966,12831,14963,12804,14954,12723,14936,12564,14917,12347,14900,11958,14886,11569,14878,11247,14859,10765,14828,10401,14784,10011,14727,9600,14660,9289,14586,8893,14508,8533,15111,12234,15110,12234,15104,12216,15092,12156,15067,12010,15028,11776,14981,11500,14942,11205,14902,10752,14861,10393,14812,9991,14752,9570,14682,9252,14603,8808,14519,8445,14431,8145,15209,11449,15208,11451,15202,11451,15190,11438,15163,11384,15117,11274,15055,10979,14994,10648,14932,10343,14871,9936,14803,9532,14729,9218,14645,8742,14556,8381,14461,8020,14365,7603,15273,10603,15272,10607,15267,10619,15256,10631,15231,10614,15182,10535,15118,10389,15042,10167,14963,9787,14883,9447,14800,9115,14710,8665,14615,8318,14514,7911,14411,7507,14279,7198,15314,9675,15313,9683,15309,9712,15298,9759,15277,9797,15229,9773,15166,9668,15084,9487,14995,9274,14898,8910,14800,8539,14697,8234,14590,7790,14479,7409,14367,7067,14178,6621,15337,8619,15337,8631,15333,8677,15325,8769,15305,8871,15264,8940,15202,8909,15119,8775,15022,8565,14916,8328,14804,8009,14688,7614,14569,7287,14448,6888,14321,6483,14088,6171,15350,7402,15350,7419,15347,7480,15340,7613,15322,7804,15287,7973,15229,8057,15148,8012,15046,7846,14933,7611,14810,7357,14682,7069,14552,6656,14421,6316,14251,5948,14007,5528,15356,5942,15356,5977,15353,6119,15348,6294,15332,6551,15302,6824,15249,7044,15171,7122,15070,7050,14949,6861,14818,6611,14679,6349,14538,6067,14398,5651,14189,5311,13935,4958,15359,4123,15359,4153,15356,4296,15353,4646,15338,5160,15311,5508,15263,5829,15188,6042,15088,6094,14966,6001,14826,5796,14678,5543,14527,5287,14377,4985,14133,4586,13869,4257,15360,1563,15360,1642,15358,2076,15354,2636,15341,3350,15317,4019,15273,4429,15203,4732,15105,4911,14981,4932,14836,4818,14679,4621,14517,4386,14359,4156,14083,3795,13808,3437,15360,122,15360,137,15358,285,15355,636,15344,1274,15322,2177,15281,2765,15215,3223,15120,3451,14995,3569,14846,3567,14681,3466,14511,3305,14344,3121,14037,2800,13753,2467,15360,0,15360,1,15359,21,15355,89,15346,253,15325,479,15287,796,15225,1148,15133,1492,15008,1749,14856,1882,14685,1886,14506,1783,14324,1608,13996,1398,13702,1183]);let An=null;function x_(){return An===null&&(An=new ff(v_,16,16,Wi,Yn),An.name="DFG_LUT",An.minFilter=Qt,An.magFilter=Qt,An.wrapS=Xn,An.wrapT=Xn,An.generateMipmaps=!1,An.needsUpdate=!0),An}class M_{constructor(e={}){const{canvas:t=Nu(),context:n=null,depth:r=!0,stencil:s=!1,alpha:a=!1,antialias:o=!1,premultipliedAlpha:u=!0,preserveDrawingBuffer:l=!1,powerPreference:h="default",failIfMajorPerformanceCaveat:p=!1,reversedDepthBuffer:m=!1,outputBufferType:v=dn}=e;this.isWebGLRenderer=!0;let y;if(n!==null){if(typeof WebGLRenderingContext<"u"&&n instanceof WebGLRenderingContext)throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");y=n.getContextAttributes().alpha}else y=a;const b=v,g=new Set([eo,Qa,Ja]),d=new Set([dn,Ln,hr,dr,ja,Za]),R=new Uint32Array(4),C=new Int32Array(4);let A=null,P=null;const D=[],I=[];let G=null;this.domElement=t,this.debug={checkShaderErrors:!0,onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this.toneMapping=Pn,this.toneMappingExposure=1,this.transmissionResolutionScale=1;const x=this;let E=!1;this._outputColorSpace=mn;let B=0,j=0,K=null,ie=-1,oe=null;const Q=new Nt,te=new Nt;let ue=null;const Ae=new yt(0);let Ee=0,Le=t.width,at=t.height,it=1,Ut=null,Dt=null;const re=new Nt(0,0,Le,at),fe=new Nt(0,0,Le,at);let Ne=!1;const Qe=new ao;let Ve=!1,ht=!1;const Vt=new Ot,Je=new $,ut=new Nt,Tt={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0};let nt=!1;function Ft(){return K===null?it:1}let F=n;function It(M,z){return t.getContext(M,z)}try{const M={alpha:!0,depth:r,stencil:s,antialias:o,premultipliedAlpha:u,preserveDrawingBuffer:l,powerPreference:h,failIfMajorPerformanceCaveat:p};if("setAttribute"in t&&t.setAttribute("data-engine",`three.js r${qa}`),t.addEventListener("webglcontextlost",Ke,!1),t.addEventListener("webglcontextrestored",At,!1),t.addEventListener("webglcontextcreationerror",vt,!1),F===null){const z="webgl2";if(F=It(z,M),F===null)throw It(z)?new Error("Error creating WebGL context with your selected attributes."):new Error("Error creating WebGL context.")}}catch(M){throw xt("WebGLRenderer: "+M.message),M}let mt,gt,Oe,w,_,V,ne,se,ee,Me,ve,me,Ye,ce,Se,Be,Ge,ye,tt,O,we,_e,De,de;function le(){mt=new xp(F),mt.init(),_e=new u_(F,mt),gt=new up(F,mt,e,_e),Oe=new l_(F,mt),gt.reversedDepthBuffer&&m&&Oe.buffers.depth.setReversed(!0),w=new yp(F),_=new qm,V=new c_(F,mt,Oe,_,gt,_e,w),ne=new hp(x),se=new vp(x),ee=new Af(F),De=new lp(F,ee),Me=new Mp(F,ee,w,De),ve=new bp(F,Me,ee,w),tt=new Ep(F,gt,V),Be=new fp(_),me=new $m(x,ne,se,mt,gt,De,Be),Ye=new __(x,_),ce=new Km,Se=new t_(mt),ye=new op(x,ne,se,Oe,ve,y,u),Ge=new a_(x,ve,gt),de=new g_(F,w,gt,Oe),O=new cp(F,mt,w),we=new Sp(F,mt,w),w.programs=me.programs,x.capabilities=gt,x.extensions=mt,x.properties=_,x.renderLists=ce,x.shadowMap=Ge,x.state=Oe,x.info=w}le(),b!==dn&&(G=new Ap(b,t.width,t.height,r,s));const xe=new p_(x,F);this.xr=xe,this.getContext=function(){return F},this.getContextAttributes=function(){return F.getContextAttributes()},this.forceContextLoss=function(){const M=mt.get("WEBGL_lose_context");M&&M.loseContext()},this.forceContextRestore=function(){const M=mt.get("WEBGL_lose_context");M&&M.restoreContext()},this.getPixelRatio=function(){return it},this.setPixelRatio=function(M){M!==void 0&&(it=M,this.setSize(Le,at,!1))},this.getSize=function(M){return M.set(Le,at)},this.setSize=function(M,z,Y=!0){if(xe.isPresenting){et("WebGLRenderer: Can't change size while VR device is presenting.");return}Le=M,at=z,t.width=Math.floor(M*it),t.height=Math.floor(z*it),Y===!0&&(t.style.width=M+"px",t.style.height=z+"px"),G!==null&&G.setSize(t.width,t.height),this.setViewport(0,0,M,z)},this.getDrawingBufferSize=function(M){return M.set(Le*it,at*it).floor()},this.setDrawingBufferSize=function(M,z,Y){Le=M,at=z,it=Y,t.width=Math.floor(M*Y),t.height=Math.floor(z*Y),this.setViewport(0,0,M,z)},this.setEffects=function(M){if(b===dn){console.error("THREE.WebGLRenderer: setEffects() requires outputBufferType set to HalfFloatType or FloatType.");return}if(M){for(let z=0;z<M.length;z++)if(M[z].isOutputPass===!0){console.warn("THREE.WebGLRenderer: OutputPass is not needed in setEffects(). Tone mapping and color space conversion are applied automatically.");break}}G.setEffects(M||[])},this.getCurrentViewport=function(M){return M.copy(Q)},this.getViewport=function(M){return M.copy(re)},this.setViewport=function(M,z,Y,q){M.isVector4?re.set(M.x,M.y,M.z,M.w):re.set(M,z,Y,q),Oe.viewport(Q.copy(re).multiplyScalar(it).round())},this.getScissor=function(M){return M.copy(fe)},this.setScissor=function(M,z,Y,q){M.isVector4?fe.set(M.x,M.y,M.z,M.w):fe.set(M,z,Y,q),Oe.scissor(te.copy(fe).multiplyScalar(it).round())},this.getScissorTest=function(){return Ne},this.setScissorTest=function(M){Oe.setScissorTest(Ne=M)},this.setOpaqueSort=function(M){Ut=M},this.setTransparentSort=function(M){Dt=M},this.getClearColor=function(M){return M.copy(ye.getClearColor())},this.setClearColor=function(){ye.setClearColor(...arguments)},this.getClearAlpha=function(){return ye.getClearAlpha()},this.setClearAlpha=function(){ye.setClearAlpha(...arguments)},this.clear=function(M=!0,z=!0,Y=!0){let q=0;if(M){let W=!1;if(K!==null){const be=K.texture.format;W=g.has(be)}if(W){const be=K.texture.type,Ie=d.has(be),Re=ye.getClearColor(),Fe=ye.getClearAlpha(),He=Re.r,$e=Re.g,We=Re.b;Ie?(R[0]=He,R[1]=$e,R[2]=We,R[3]=Fe,F.clearBufferuiv(F.COLOR,0,R)):(C[0]=He,C[1]=$e,C[2]=We,C[3]=Fe,F.clearBufferiv(F.COLOR,0,C))}else q|=F.COLOR_BUFFER_BIT}z&&(q|=F.DEPTH_BUFFER_BIT),Y&&(q|=F.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),F.clear(q)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.dispose=function(){t.removeEventListener("webglcontextlost",Ke,!1),t.removeEventListener("webglcontextrestored",At,!1),t.removeEventListener("webglcontextcreationerror",vt,!1),ye.dispose(),ce.dispose(),Se.dispose(),_.dispose(),ne.dispose(),se.dispose(),ve.dispose(),De.dispose(),de.dispose(),me.dispose(),xe.dispose(),xe.removeEventListener("sessionstart",yi),xe.removeEventListener("sessionend",Sr),Nn.stop()};function Ke(M){M.preventDefault(),Po("WebGLRenderer: Context Lost."),E=!0}function At(){Po("WebGLRenderer: Context Restored."),E=!1;const M=w.autoReset,z=Ge.enabled,Y=Ge.autoUpdate,q=Ge.needsUpdate,W=Ge.type;le(),w.autoReset=M,Ge.enabled=z,Ge.autoUpdate=Y,Ge.needsUpdate=q,Ge.type=W}function vt(M){xt("WebGLRenderer: A WebGL context could not be created. Reason: ",M.statusMessage)}function $t(M){const z=M.target;z.removeEventListener("dispose",$t),vn(z)}function vn(M){ds(M),_.remove(M)}function ds(M){const z=_.get(M).programs;z!==void 0&&(z.forEach(function(Y){me.releaseProgram(Y)}),M.isShaderMaterial&&me.releaseShaderCache(M))}this.renderBufferDirect=function(M,z,Y,q,W,be){z===null&&(z=Tt);const Ie=W.isMesh&&W.matrixWorld.determinant()<0,Re=bi(M,z,Y,q,W);Oe.setMaterial(q,Ie);let Fe=Y.index,He=1;if(q.wireframe===!0){if(Fe=Me.getWireframeAttribute(Y),Fe===void 0)return;He=2}const $e=Y.drawRange,We=Y.attributes.position;let ot=$e.start*He,Mt=($e.start+$e.count)*He;be!==null&&(ot=Math.max(ot,be.start*He),Mt=Math.min(Mt,(be.start+be.count)*He)),Fe!==null?(ot=Math.max(ot,0),Mt=Math.min(Mt,Fe.count)):We!=null&&(ot=Math.max(ot,0),Mt=Math.min(Mt,We.count));const Et=Mt-ot;if(Et<0||Et===1/0)return;De.setup(W,q,Re,Y,Fe);let Lt,je=O;if(Fe!==null&&(Lt=ee.get(Fe),je=we,je.setIndex(Lt)),W.isMesh)q.wireframe===!0?(Oe.setLineWidth(q.wireframeLinewidth*Ft()),je.setMode(F.LINES)):je.setMode(F.TRIANGLES);else if(W.isLine){let ke=q.linewidth;ke===void 0&&(ke=1),Oe.setLineWidth(ke*Ft()),W.isLineSegments?je.setMode(F.LINES):W.isLineLoop?je.setMode(F.LINE_LOOP):je.setMode(F.LINE_STRIP)}else W.isPoints?je.setMode(F.POINTS):W.isSprite&&je.setMode(F.TRIANGLES);if(W.isBatchedMesh)if(W._multiDrawInstances!==null)pr("WebGLRenderer: renderMultiDrawInstances has been deprecated and will be removed in r184. Append to renderMultiDraw arguments and use indirection."),je.renderMultiDrawInstances(W._multiDrawStarts,W._multiDrawCounts,W._multiDrawCount,W._multiDrawInstances);else if(mt.get("WEBGL_multi_draw"))je.renderMultiDraw(W._multiDrawStarts,W._multiDrawCounts,W._multiDrawCount);else{const ke=W._multiDrawStarts,St=W._multiDrawCounts,ft=W._multiDrawCount,en=Fe?ee.get(Fe).bytesPerElement:1,jn=_.get(q).currentProgram.getUniforms();for(let qt=0;qt<ft;qt++)jn.setValue(F,"_gl_DrawID",qt),je.render(ke[qt]/en,St[qt])}else if(W.isInstancedMesh)je.renderInstances(ot,Et,W.count);else if(Y.isInstancedBufferGeometry){const ke=Y._maxInstanceCount!==void 0?Y._maxInstanceCount:1/0,St=Math.min(Y.instanceCount,ke);je.renderInstances(ot,Et,St)}else je.render(ot,Et)};function Mr(M,z,Y){M.transparent===!0&&M.side===Wn&&M.forceSinglePass===!1?(M.side=on,M.needsUpdate=!0,On(M,z,Y),M.side=ai,M.needsUpdate=!0,On(M,z,Y),M.side=Wn):On(M,z,Y)}this.compile=function(M,z,Y=null){Y===null&&(Y=M),P=Se.get(Y),P.init(z),I.push(P),Y.traverseVisible(function(W){W.isLight&&W.layers.test(z.layers)&&(P.pushLight(W),W.castShadow&&P.pushShadow(W))}),M!==Y&&M.traverseVisible(function(W){W.isLight&&W.layers.test(z.layers)&&(P.pushLight(W),W.castShadow&&P.pushShadow(W))}),P.setupLights();const q=new Set;return M.traverse(function(W){if(!(W.isMesh||W.isPoints||W.isLine||W.isSprite))return;const be=W.material;if(be)if(Array.isArray(be))for(let Ie=0;Ie<be.length;Ie++){const Re=be[Ie];Mr(Re,Y,W),q.add(Re)}else Mr(be,Y,W),q.add(be)}),P=I.pop(),q},this.compileAsync=function(M,z,Y=null){const q=this.compile(M,z,Y);return new Promise(W=>{function be(){if(q.forEach(function(Ie){_.get(Ie).currentProgram.isReady()&&q.delete(Ie)}),q.size===0){W(M);return}setTimeout(be,10)}mt.get("KHR_parallel_shader_compile")!==null?be():setTimeout(be,10)})};let Zi=null;function Ji(M){Zi&&Zi(M)}function yi(){Nn.stop()}function Sr(){Nn.start()}const Nn=new jl;Nn.setAnimationLoop(Ji),typeof self<"u"&&Nn.setContext(self),this.setAnimationLoop=function(M){Zi=M,xe.setAnimationLoop(M),M===null?Nn.stop():Nn.start()},xe.addEventListener("sessionstart",yi),xe.addEventListener("sessionend",Sr),this.render=function(M,z){if(z!==void 0&&z.isCamera!==!0){xt("WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(E===!0)return;const Y=xe.enabled===!0&&xe.isPresenting===!0,q=G!==null&&(K===null||Y)&&G.begin(x,K);if(M.matrixWorldAutoUpdate===!0&&M.updateMatrixWorld(),z.parent===null&&z.matrixWorldAutoUpdate===!0&&z.updateMatrixWorld(),xe.enabled===!0&&xe.isPresenting===!0&&(G===null||G.isCompositing()===!1)&&(xe.cameraAutoUpdate===!0&&xe.updateCamera(z),z=xe.getCamera()),M.isScene===!0&&M.onBeforeRender(x,M,z,K),P=Se.get(M,I.length),P.init(z),I.push(P),Vt.multiplyMatrices(z.projectionMatrix,z.matrixWorldInverse),Qe.setFromProjectionMatrix(Vt,Cn,z.reversedDepth),ht=this.localClippingEnabled,Ve=Be.init(this.clippingPlanes,ht),A=ce.get(M,D.length),A.init(),D.push(A),xe.enabled===!0&&xe.isPresenting===!0){const Ie=x.xr.getDepthSensingMesh();Ie!==null&&Qi(Ie,z,-1/0,x.sortObjects)}Qi(M,z,0,x.sortObjects),A.finish(),x.sortObjects===!0&&A.sort(Ut,Dt),nt=xe.enabled===!1||xe.isPresenting===!1||xe.hasDepthSensing()===!1,nt&&ye.addToRenderList(A,M),this.info.render.frame++,Ve===!0&&Be.beginShadows();const W=P.state.shadowsArray;if(Ge.render(W,M,z),Ve===!0&&Be.endShadows(),this.info.autoReset===!0&&this.info.reset(),(q&&G.hasRenderPass())===!1){const Ie=A.opaque,Re=A.transmissive;if(P.setupLights(),z.isArrayCamera){const Fe=z.cameras;if(Re.length>0)for(let He=0,$e=Fe.length;He<$e;He++){const We=Fe[He];yr(Ie,Re,M,We)}nt&&ye.render(M);for(let He=0,$e=Fe.length;He<$e;He++){const We=Fe[He];er(A,M,We,We.viewport)}}else Re.length>0&&yr(Ie,Re,M,z),nt&&ye.render(M),er(A,M,z)}K!==null&&j===0&&(V.updateMultisampleRenderTarget(K),V.updateRenderTargetMipmap(K)),q&&G.end(x),M.isScene===!0&&M.onAfterRender(x,M,z),De.resetDefaultState(),ie=-1,oe=null,I.pop(),I.length>0?(P=I[I.length-1],Ve===!0&&Be.setGlobalState(x.clippingPlanes,P.state.camera)):P=null,D.pop(),D.length>0?A=D[D.length-1]:A=null};function Qi(M,z,Y,q){if(M.visible===!1)return;if(M.layers.test(z.layers)){if(M.isGroup)Y=M.renderOrder;else if(M.isLOD)M.autoUpdate===!0&&M.update(z);else if(M.isLight)P.pushLight(M),M.castShadow&&P.pushShadow(M);else if(M.isSprite){if(!M.frustumCulled||Qe.intersectsSprite(M)){q&&ut.setFromMatrixPosition(M.matrixWorld).applyMatrix4(Vt);const Ie=ve.update(M),Re=M.material;Re.visible&&A.push(M,Ie,Re,Y,ut.z,null)}}else if((M.isMesh||M.isLine||M.isPoints)&&(!M.frustumCulled||Qe.intersectsObject(M))){const Ie=ve.update(M),Re=M.material;if(q&&(M.boundingSphere!==void 0?(M.boundingSphere===null&&M.computeBoundingSphere(),ut.copy(M.boundingSphere.center)):(Ie.boundingSphere===null&&Ie.computeBoundingSphere(),ut.copy(Ie.boundingSphere.center)),ut.applyMatrix4(M.matrixWorld).applyMatrix4(Vt)),Array.isArray(Re)){const Fe=Ie.groups;for(let He=0,$e=Fe.length;He<$e;He++){const We=Fe[He],ot=Re[We.materialIndex];ot&&ot.visible&&A.push(M,Ie,ot,Y,ut.z,We)}}else Re.visible&&A.push(M,Ie,Re,Y,ut.z,null)}}const be=M.children;for(let Ie=0,Re=be.length;Ie<Re;Ie++)Qi(be[Ie],z,Y,q)}function er(M,z,Y,q){const{opaque:W,transmissive:be,transparent:Ie}=M;P.setupLightsView(Y),Ve===!0&&Be.setGlobalState(x.clippingPlanes,Y),q&&Oe.viewport(Q.copy(q)),W.length>0&&Ei(W,z,Y),be.length>0&&Ei(be,z,Y),Ie.length>0&&Ei(Ie,z,Y),Oe.buffers.depth.setTest(!0),Oe.buffers.depth.setMask(!0),Oe.buffers.color.setMask(!0),Oe.setPolygonOffset(!1)}function yr(M,z,Y,q){if((Y.isScene===!0?Y.overrideMaterial:null)!==null)return;if(P.state.transmissionRenderTarget[q.id]===void 0){const ot=mt.has("EXT_color_buffer_half_float")||mt.has("EXT_color_buffer_float");P.state.transmissionRenderTarget[q.id]=new Dn(1,1,{generateMipmaps:!0,type:ot?Yn:dn,minFilter:xi,samples:gt.samples,stencilBuffer:s,resolveDepthBuffer:!1,resolveStencilBuffer:!1,colorSpace:_t.workingColorSpace})}const be=P.state.transmissionRenderTarget[q.id],Ie=q.viewport||Q;be.setSize(Ie.z*x.transmissionResolutionScale,Ie.w*x.transmissionResolutionScale);const Re=x.getRenderTarget(),Fe=x.getActiveCubeFace(),He=x.getActiveMipmapLevel();x.setRenderTarget(be),x.getClearColor(Ae),Ee=x.getClearAlpha(),Ee<1&&x.setClearColor(16777215,.5),x.clear(),nt&&ye.render(Y);const $e=x.toneMapping;x.toneMapping=Pn;const We=q.viewport;if(q.viewport!==void 0&&(q.viewport=void 0),P.setupLightsView(q),Ve===!0&&Be.setGlobalState(x.clippingPlanes,q),Ei(M,Y,q),V.updateMultisampleRenderTarget(be),V.updateRenderTargetMipmap(be),mt.has("WEBGL_multisampled_render_to_texture")===!1){let ot=!1;for(let Mt=0,Et=z.length;Mt<Et;Mt++){const Lt=z[Mt],{object:je,geometry:ke,material:St,group:ft}=Lt;if(St.side===Wn&&je.layers.test(q.layers)){const en=St.side;St.side=on,St.needsUpdate=!0,Er(je,Y,q,ke,St,ft),St.side=en,St.needsUpdate=!0,ot=!0}}ot===!0&&(V.updateMultisampleRenderTarget(be),V.updateRenderTargetMipmap(be))}x.setRenderTarget(Re,Fe,He),x.setClearColor(Ae,Ee),We!==void 0&&(q.viewport=We),x.toneMapping=$e}function Ei(M,z,Y){const q=z.isScene===!0?z.overrideMaterial:null;for(let W=0,be=M.length;W<be;W++){const Ie=M[W],{object:Re,geometry:Fe,group:He}=Ie;let $e=Ie.material;$e.allowOverride===!0&&q!==null&&($e=q),Re.layers.test(Y.layers)&&Er(Re,z,Y,Fe,$e,He)}}function Er(M,z,Y,q,W,be){M.onBeforeRender(x,z,Y,q,W,be),M.modelViewMatrix.multiplyMatrices(Y.matrixWorldInverse,M.matrixWorld),M.normalMatrix.getNormalMatrix(M.modelViewMatrix),W.onBeforeRender(x,z,Y,q,M,be),W.transparent===!0&&W.side===Wn&&W.forceSinglePass===!1?(W.side=on,W.needsUpdate=!0,x.renderBufferDirect(Y,z,q,W,M,be),W.side=ai,W.needsUpdate=!0,x.renderBufferDirect(Y,z,q,W,M,be),W.side=Wn):x.renderBufferDirect(Y,z,q,W,M,be),M.onAfterRender(x,z,Y,q,W,be)}function On(M,z,Y){z.isScene!==!0&&(z=Tt);const q=_.get(M),W=P.state.lights,be=P.state.shadowsArray,Ie=W.state.version,Re=me.getParameters(M,W.state,be,z,Y),Fe=me.getProgramCacheKey(Re);let He=q.programs;q.environment=M.isMeshStandardMaterial?z.environment:null,q.fog=z.fog,q.envMap=(M.isMeshStandardMaterial?se:ne).get(M.envMap||q.environment),q.envMapRotation=q.environment!==null&&M.envMap===null?z.environmentRotation:M.envMapRotation,He===void 0&&(M.addEventListener("dispose",$t),He=new Map,q.programs=He);let $e=He.get(Fe);if($e!==void 0){if(q.currentProgram===$e&&q.lightsStateVersion===Ie)return Tr(M,Re),$e}else Re.uniforms=me.getUniforms(M),M.onBeforeCompile(Re,x),$e=me.acquireProgram(Re,Fe),He.set(Fe,$e),q.uniforms=Re.uniforms;const We=q.uniforms;return(!M.isShaderMaterial&&!M.isRawShaderMaterial||M.clipping===!0)&&(We.clippingPlanes=Be.uniform),Tr(M,Re),q.needsLights=Ti(M),q.lightsStateVersion=Ie,q.needsLights&&(We.ambientLightColor.value=W.state.ambient,We.lightProbe.value=W.state.probe,We.directionalLights.value=W.state.directional,We.directionalLightShadows.value=W.state.directionalShadow,We.spotLights.value=W.state.spot,We.spotLightShadows.value=W.state.spotShadow,We.rectAreaLights.value=W.state.rectArea,We.ltc_1.value=W.state.rectAreaLTC1,We.ltc_2.value=W.state.rectAreaLTC2,We.pointLights.value=W.state.point,We.pointLightShadows.value=W.state.pointShadow,We.hemisphereLights.value=W.state.hemi,We.directionalShadowMap.value=W.state.directionalShadowMap,We.directionalShadowMatrix.value=W.state.directionalShadowMatrix,We.spotShadowMap.value=W.state.spotShadowMap,We.spotLightMatrix.value=W.state.spotLightMatrix,We.spotLightMap.value=W.state.spotLightMap,We.pointShadowMap.value=W.state.pointShadowMap,We.pointShadowMatrix.value=W.state.pointShadowMatrix),q.currentProgram=$e,q.uniformsList=null,$e}function br(M){if(M.uniformsList===null){const z=M.currentProgram.getUniforms();M.uniformsList=ts.seqWithValue(z.seq,M.uniforms)}return M.uniformsList}function Tr(M,z){const Y=_.get(M);Y.outputColorSpace=z.outputColorSpace,Y.batching=z.batching,Y.batchingColor=z.batchingColor,Y.instancing=z.instancing,Y.instancingColor=z.instancingColor,Y.instancingMorph=z.instancingMorph,Y.skinning=z.skinning,Y.morphTargets=z.morphTargets,Y.morphNormals=z.morphNormals,Y.morphColors=z.morphColors,Y.morphTargetsCount=z.morphTargetsCount,Y.numClippingPlanes=z.numClippingPlanes,Y.numIntersection=z.numClipIntersection,Y.vertexAlphas=z.vertexAlphas,Y.vertexTangents=z.vertexTangents,Y.toneMapping=z.toneMapping}function bi(M,z,Y,q,W){z.isScene!==!0&&(z=Tt),V.resetTextureUnits();const be=z.fog,Ie=q.isMeshStandardMaterial?z.environment:null,Re=K===null?x.outputColorSpace:K.isXRRenderTarget===!0?K.texture.colorSpace:Xi,Fe=(q.isMeshStandardMaterial?se:ne).get(q.envMap||Ie),He=q.vertexColors===!0&&!!Y.attributes.color&&Y.attributes.color.itemSize===4,$e=!!Y.attributes.tangent&&(!!q.normalMap||q.anisotropy>0),We=!!Y.morphAttributes.position,ot=!!Y.morphAttributes.normal,Mt=!!Y.morphAttributes.color;let Et=Pn;q.toneMapped&&(K===null||K.isXRRenderTarget===!0)&&(Et=x.toneMapping);const Lt=Y.morphAttributes.position||Y.morphAttributes.normal||Y.morphAttributes.color,je=Lt!==void 0?Lt.length:0,ke=_.get(q),St=P.state.lights;if(Ve===!0&&(ht===!0||M!==oe)){const Wt=M===oe&&q.id===ie;Be.setState(q,M,Wt)}let ft=!1;q.version===ke.__version?(ke.needsLights&&ke.lightsStateVersion!==St.state.version||ke.outputColorSpace!==Re||W.isBatchedMesh&&ke.batching===!1||!W.isBatchedMesh&&ke.batching===!0||W.isBatchedMesh&&ke.batchingColor===!0&&W.colorTexture===null||W.isBatchedMesh&&ke.batchingColor===!1&&W.colorTexture!==null||W.isInstancedMesh&&ke.instancing===!1||!W.isInstancedMesh&&ke.instancing===!0||W.isSkinnedMesh&&ke.skinning===!1||!W.isSkinnedMesh&&ke.skinning===!0||W.isInstancedMesh&&ke.instancingColor===!0&&W.instanceColor===null||W.isInstancedMesh&&ke.instancingColor===!1&&W.instanceColor!==null||W.isInstancedMesh&&ke.instancingMorph===!0&&W.morphTexture===null||W.isInstancedMesh&&ke.instancingMorph===!1&&W.morphTexture!==null||ke.envMap!==Fe||q.fog===!0&&ke.fog!==be||ke.numClippingPlanes!==void 0&&(ke.numClippingPlanes!==Be.numPlanes||ke.numIntersection!==Be.numIntersection)||ke.vertexAlphas!==He||ke.vertexTangents!==$e||ke.morphTargets!==We||ke.morphNormals!==ot||ke.morphColors!==Mt||ke.toneMapping!==Et||ke.morphTargetsCount!==je)&&(ft=!0):(ft=!0,ke.__version=q.version);let en=ke.currentProgram;ft===!0&&(en=On(q,z,W));let jn=!1,qt=!1,oi=!1;const Rt=en.getUniforms(),Yt=ke.uniforms;if(Oe.useProgram(en.program)&&(jn=!0,qt=!0,oi=!0),q.id!==ie&&(ie=q.id,qt=!0),jn||oe!==M){Oe.buffers.depth.getReversed()&&M.reversedDepth!==!0&&(M._reversedDepth=!0,M.updateProjectionMatrix()),Rt.setValue(F,"projectionMatrix",M.projectionMatrix),Rt.setValue(F,"viewMatrix",M.matrixWorldInverse);const Kt=Rt.map.cameraPosition;Kt!==void 0&&Kt.setValue(F,Je.setFromMatrixPosition(M.matrixWorld)),gt.logarithmicDepthBuffer&&Rt.setValue(F,"logDepthBufFC",2/(Math.log(M.far+1)/Math.LN2)),(q.isMeshPhongMaterial||q.isMeshToonMaterial||q.isMeshLambertMaterial||q.isMeshBasicMaterial||q.isMeshStandardMaterial||q.isShaderMaterial)&&Rt.setValue(F,"isOrthographic",M.isOrthographicCamera===!0),oe!==M&&(oe=M,qt=!0,oi=!0)}if(ke.needsLights&&(St.state.directionalShadowMap.length>0&&Rt.setValue(F,"directionalShadowMap",St.state.directionalShadowMap,V),St.state.spotShadowMap.length>0&&Rt.setValue(F,"spotShadowMap",St.state.spotShadowMap,V),St.state.pointShadowMap.length>0&&Rt.setValue(F,"pointShadowMap",St.state.pointShadowMap,V)),W.isSkinnedMesh){Rt.setOptional(F,W,"bindMatrix"),Rt.setOptional(F,W,"bindMatrixInverse");const Wt=W.skeleton;Wt&&(Wt.boneTexture===null&&Wt.computeBoneTexture(),Rt.setValue(F,"boneTexture",Wt.boneTexture,V))}W.isBatchedMesh&&(Rt.setOptional(F,W,"batchingTexture"),Rt.setValue(F,"batchingTexture",W._matricesTexture,V),Rt.setOptional(F,W,"batchingIdTexture"),Rt.setValue(F,"batchingIdTexture",W._indirectTexture,V),Rt.setOptional(F,W,"batchingColorTexture"),W._colorsTexture!==null&&Rt.setValue(F,"batchingColorTexture",W._colorsTexture,V));const sn=Y.morphAttributes;if((sn.position!==void 0||sn.normal!==void 0||sn.color!==void 0)&&tt.update(W,Y,en),(qt||ke.receiveShadow!==W.receiveShadow)&&(ke.receiveShadow=W.receiveShadow,Rt.setValue(F,"receiveShadow",W.receiveShadow)),q.isMeshGouraudMaterial&&q.envMap!==null&&(Yt.envMap.value=Fe,Yt.flipEnvMap.value=Fe.isCubeTexture&&Fe.isRenderTargetTexture===!1?-1:1),q.isMeshStandardMaterial&&q.envMap===null&&z.environment!==null&&(Yt.envMapIntensity.value=z.environmentIntensity),Yt.dfgLUT!==void 0&&(Yt.dfgLUT.value=x_()),qt&&(Rt.setValue(F,"toneMappingExposure",x.toneMappingExposure),ke.needsLights&&Ar(Yt,oi),be&&q.fog===!0&&Ye.refreshFogUniforms(Yt,be),Ye.refreshMaterialUniforms(Yt,q,it,at,P.state.transmissionRenderTarget[M.id]),ts.upload(F,br(ke),Yt,V)),q.isShaderMaterial&&q.uniformsNeedUpdate===!0&&(ts.upload(F,br(ke),Yt,V),q.uniformsNeedUpdate=!1),q.isSpriteMaterial&&Rt.setValue(F,"center",W.center),Rt.setValue(F,"modelViewMatrix",W.modelViewMatrix),Rt.setValue(F,"normalMatrix",W.normalMatrix),Rt.setValue(F,"modelMatrix",W.matrixWorld),q.isShaderMaterial||q.isRawShaderMaterial){const Wt=q.uniformsGroups;for(let Kt=0,tr=Wt.length;Kt<tr;Kt++){const Bn=Wt[Kt];de.update(Bn,en),de.bind(Bn,en)}}return en}function Ar(M,z){M.ambientLightColor.needsUpdate=z,M.lightProbe.needsUpdate=z,M.directionalLights.needsUpdate=z,M.directionalLightShadows.needsUpdate=z,M.pointLights.needsUpdate=z,M.pointLightShadows.needsUpdate=z,M.spotLights.needsUpdate=z,M.spotLightShadows.needsUpdate=z,M.rectAreaLights.needsUpdate=z,M.hemisphereLights.needsUpdate=z}function Ti(M){return M.isMeshLambertMaterial||M.isMeshToonMaterial||M.isMeshPhongMaterial||M.isMeshStandardMaterial||M.isShadowMaterial||M.isShaderMaterial&&M.lights===!0}this.getActiveCubeFace=function(){return B},this.getActiveMipmapLevel=function(){return j},this.getRenderTarget=function(){return K},this.setRenderTargetTextures=function(M,z,Y){const q=_.get(M);q.__autoAllocateDepthBuffer=M.resolveDepthBuffer===!1,q.__autoAllocateDepthBuffer===!1&&(q.__useRenderToTexture=!1),_.get(M.texture).__webglTexture=z,_.get(M.depthTexture).__webglTexture=q.__autoAllocateDepthBuffer?void 0:Y,q.__hasExternalTextures=!0},this.setRenderTargetFramebuffer=function(M,z){const Y=_.get(M);Y.__webglFramebuffer=z,Y.__useDefaultFramebuffer=z===void 0};const ps=F.createFramebuffer();this.setRenderTarget=function(M,z=0,Y=0){K=M,B=z,j=Y;let q=null,W=!1,be=!1;if(M){const Re=_.get(M);if(Re.__useDefaultFramebuffer!==void 0){Oe.bindFramebuffer(F.FRAMEBUFFER,Re.__webglFramebuffer),Q.copy(M.viewport),te.copy(M.scissor),ue=M.scissorTest,Oe.viewport(Q),Oe.scissor(te),Oe.setScissorTest(ue),ie=-1;return}else if(Re.__webglFramebuffer===void 0)V.setupRenderTarget(M);else if(Re.__hasExternalTextures)V.rebindTextures(M,_.get(M.texture).__webglTexture,_.get(M.depthTexture).__webglTexture);else if(M.depthBuffer){const $e=M.depthTexture;if(Re.__boundDepthTexture!==$e){if($e!==null&&_.has($e)&&(M.width!==$e.image.width||M.height!==$e.image.height))throw new Error("WebGLRenderTarget: Attached DepthTexture is initialized to the incorrect size.");V.setupDepthRenderbuffer(M)}}const Fe=M.texture;(Fe.isData3DTexture||Fe.isDataArrayTexture||Fe.isCompressedArrayTexture)&&(be=!0);const He=_.get(M).__webglFramebuffer;M.isWebGLCubeRenderTarget?(Array.isArray(He[z])?q=He[z][Y]:q=He[z],W=!0):M.samples>0&&V.useMultisampledRTT(M)===!1?q=_.get(M).__webglMultisampledFramebuffer:Array.isArray(He)?q=He[Y]:q=He,Q.copy(M.viewport),te.copy(M.scissor),ue=M.scissorTest}else Q.copy(re).multiplyScalar(it).floor(),te.copy(fe).multiplyScalar(it).floor(),ue=Ne;if(Y!==0&&(q=ps),Oe.bindFramebuffer(F.FRAMEBUFFER,q)&&Oe.drawBuffers(M,q),Oe.viewport(Q),Oe.scissor(te),Oe.setScissorTest(ue),W){const Re=_.get(M.texture);F.framebufferTexture2D(F.FRAMEBUFFER,F.COLOR_ATTACHMENT0,F.TEXTURE_CUBE_MAP_POSITIVE_X+z,Re.__webglTexture,Y)}else if(be){const Re=z;for(let Fe=0;Fe<M.textures.length;Fe++){const He=_.get(M.textures[Fe]);F.framebufferTextureLayer(F.FRAMEBUFFER,F.COLOR_ATTACHMENT0+Fe,He.__webglTexture,Y,Re)}}else if(M!==null&&Y!==0){const Re=_.get(M.texture);F.framebufferTexture2D(F.FRAMEBUFFER,F.COLOR_ATTACHMENT0,F.TEXTURE_2D,Re.__webglTexture,Y)}ie=-1},this.readRenderTargetPixels=function(M,z,Y,q,W,be,Ie,Re=0){if(!(M&&M.isWebGLRenderTarget)){xt("WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let Fe=_.get(M).__webglFramebuffer;if(M.isWebGLCubeRenderTarget&&Ie!==void 0&&(Fe=Fe[Ie]),Fe){Oe.bindFramebuffer(F.FRAMEBUFFER,Fe);try{const He=M.textures[Re],$e=He.format,We=He.type;if(!gt.textureFormatReadable($e)){xt("WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}if(!gt.textureTypeReadable(We)){xt("WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}z>=0&&z<=M.width-q&&Y>=0&&Y<=M.height-W&&(M.textures.length>1&&F.readBuffer(F.COLOR_ATTACHMENT0+Re),F.readPixels(z,Y,q,W,_e.convert($e),_e.convert(We),be))}finally{const He=K!==null?_.get(K).__webglFramebuffer:null;Oe.bindFramebuffer(F.FRAMEBUFFER,He)}}},this.readRenderTargetPixelsAsync=async function(M,z,Y,q,W,be,Ie,Re=0){if(!(M&&M.isWebGLRenderTarget))throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");let Fe=_.get(M).__webglFramebuffer;if(M.isWebGLCubeRenderTarget&&Ie!==void 0&&(Fe=Fe[Ie]),Fe)if(z>=0&&z<=M.width-q&&Y>=0&&Y<=M.height-W){Oe.bindFramebuffer(F.FRAMEBUFFER,Fe);const He=M.textures[Re],$e=He.format,We=He.type;if(!gt.textureFormatReadable($e))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");if(!gt.textureTypeReadable(We))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");const ot=F.createBuffer();F.bindBuffer(F.PIXEL_PACK_BUFFER,ot),F.bufferData(F.PIXEL_PACK_BUFFER,be.byteLength,F.STREAM_READ),M.textures.length>1&&F.readBuffer(F.COLOR_ATTACHMENT0+Re),F.readPixels(z,Y,q,W,_e.convert($e),_e.convert(We),0);const Mt=K!==null?_.get(K).__webglFramebuffer:null;Oe.bindFramebuffer(F.FRAMEBUFFER,Mt);const Et=F.fenceSync(F.SYNC_GPU_COMMANDS_COMPLETE,0);return F.flush(),await Ou(F,Et,4),F.bindBuffer(F.PIXEL_PACK_BUFFER,ot),F.getBufferSubData(F.PIXEL_PACK_BUFFER,0,be),F.deleteBuffer(ot),F.deleteSync(Et),be}else throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.")},this.copyFramebufferToTexture=function(M,z=null,Y=0){const q=Math.pow(2,-Y),W=Math.floor(M.image.width*q),be=Math.floor(M.image.height*q),Ie=z!==null?z.x:0,Re=z!==null?z.y:0;V.setTexture2D(M,0),F.copyTexSubImage2D(F.TEXTURE_2D,Y,0,0,Ie,Re,W,be),Oe.unbindTexture()};const ms=F.createFramebuffer(),cn=F.createFramebuffer();this.copyTextureToTexture=function(M,z,Y=null,q=null,W=0,be=null){be===null&&(W!==0?(pr("WebGLRenderer: copyTextureToTexture function signature has changed to support src and dst mipmap levels."),be=W,W=0):be=0);let Ie,Re,Fe,He,$e,We,ot,Mt,Et;const Lt=M.isCompressedTexture?M.mipmaps[be]:M.image;if(Y!==null)Ie=Y.max.x-Y.min.x,Re=Y.max.y-Y.min.y,Fe=Y.isBox3?Y.max.z-Y.min.z:1,He=Y.min.x,$e=Y.min.y,We=Y.isBox3?Y.min.z:0;else{const sn=Math.pow(2,-W);Ie=Math.floor(Lt.width*sn),Re=Math.floor(Lt.height*sn),M.isDataArrayTexture?Fe=Lt.depth:M.isData3DTexture?Fe=Math.floor(Lt.depth*sn):Fe=1,He=0,$e=0,We=0}q!==null?(ot=q.x,Mt=q.y,Et=q.z):(ot=0,Mt=0,Et=0);const je=_e.convert(z.format),ke=_e.convert(z.type);let St;z.isData3DTexture?(V.setTexture3D(z,0),St=F.TEXTURE_3D):z.isDataArrayTexture||z.isCompressedArrayTexture?(V.setTexture2DArray(z,0),St=F.TEXTURE_2D_ARRAY):(V.setTexture2D(z,0),St=F.TEXTURE_2D),F.pixelStorei(F.UNPACK_FLIP_Y_WEBGL,z.flipY),F.pixelStorei(F.UNPACK_PREMULTIPLY_ALPHA_WEBGL,z.premultiplyAlpha),F.pixelStorei(F.UNPACK_ALIGNMENT,z.unpackAlignment);const ft=F.getParameter(F.UNPACK_ROW_LENGTH),en=F.getParameter(F.UNPACK_IMAGE_HEIGHT),jn=F.getParameter(F.UNPACK_SKIP_PIXELS),qt=F.getParameter(F.UNPACK_SKIP_ROWS),oi=F.getParameter(F.UNPACK_SKIP_IMAGES);F.pixelStorei(F.UNPACK_ROW_LENGTH,Lt.width),F.pixelStorei(F.UNPACK_IMAGE_HEIGHT,Lt.height),F.pixelStorei(F.UNPACK_SKIP_PIXELS,He),F.pixelStorei(F.UNPACK_SKIP_ROWS,$e),F.pixelStorei(F.UNPACK_SKIP_IMAGES,We);const Rt=M.isDataArrayTexture||M.isData3DTexture,Yt=z.isDataArrayTexture||z.isData3DTexture;if(M.isDepthTexture){const sn=_.get(M),Wt=_.get(z),Kt=_.get(sn.__renderTarget),tr=_.get(Wt.__renderTarget);Oe.bindFramebuffer(F.READ_FRAMEBUFFER,Kt.__webglFramebuffer),Oe.bindFramebuffer(F.DRAW_FRAMEBUFFER,tr.__webglFramebuffer);for(let Bn=0;Bn<Fe;Bn++)Rt&&(F.framebufferTextureLayer(F.READ_FRAMEBUFFER,F.COLOR_ATTACHMENT0,_.get(M).__webglTexture,W,We+Bn),F.framebufferTextureLayer(F.DRAW_FRAMEBUFFER,F.COLOR_ATTACHMENT0,_.get(z).__webglTexture,be,Et+Bn)),F.blitFramebuffer(He,$e,Ie,Re,ot,Mt,Ie,Re,F.DEPTH_BUFFER_BIT,F.NEAREST);Oe.bindFramebuffer(F.READ_FRAMEBUFFER,null),Oe.bindFramebuffer(F.DRAW_FRAMEBUFFER,null)}else if(W!==0||M.isRenderTargetTexture||_.has(M)){const sn=_.get(M),Wt=_.get(z);Oe.bindFramebuffer(F.READ_FRAMEBUFFER,ms),Oe.bindFramebuffer(F.DRAW_FRAMEBUFFER,cn);for(let Kt=0;Kt<Fe;Kt++)Rt?F.framebufferTextureLayer(F.READ_FRAMEBUFFER,F.COLOR_ATTACHMENT0,sn.__webglTexture,W,We+Kt):F.framebufferTexture2D(F.READ_FRAMEBUFFER,F.COLOR_ATTACHMENT0,F.TEXTURE_2D,sn.__webglTexture,W),Yt?F.framebufferTextureLayer(F.DRAW_FRAMEBUFFER,F.COLOR_ATTACHMENT0,Wt.__webglTexture,be,Et+Kt):F.framebufferTexture2D(F.DRAW_FRAMEBUFFER,F.COLOR_ATTACHMENT0,F.TEXTURE_2D,Wt.__webglTexture,be),W!==0?F.blitFramebuffer(He,$e,Ie,Re,ot,Mt,Ie,Re,F.COLOR_BUFFER_BIT,F.NEAREST):Yt?F.copyTexSubImage3D(St,be,ot,Mt,Et+Kt,He,$e,Ie,Re):F.copyTexSubImage2D(St,be,ot,Mt,He,$e,Ie,Re);Oe.bindFramebuffer(F.READ_FRAMEBUFFER,null),Oe.bindFramebuffer(F.DRAW_FRAMEBUFFER,null)}else Yt?M.isDataTexture||M.isData3DTexture?F.texSubImage3D(St,be,ot,Mt,Et,Ie,Re,Fe,je,ke,Lt.data):z.isCompressedArrayTexture?F.compressedTexSubImage3D(St,be,ot,Mt,Et,Ie,Re,Fe,je,Lt.data):F.texSubImage3D(St,be,ot,Mt,Et,Ie,Re,Fe,je,ke,Lt):M.isDataTexture?F.texSubImage2D(F.TEXTURE_2D,be,ot,Mt,Ie,Re,je,ke,Lt.data):M.isCompressedTexture?F.compressedTexSubImage2D(F.TEXTURE_2D,be,ot,Mt,Lt.width,Lt.height,je,Lt.data):F.texSubImage2D(F.TEXTURE_2D,be,ot,Mt,Ie,Re,je,ke,Lt);F.pixelStorei(F.UNPACK_ROW_LENGTH,ft),F.pixelStorei(F.UNPACK_IMAGE_HEIGHT,en),F.pixelStorei(F.UNPACK_SKIP_PIXELS,jn),F.pixelStorei(F.UNPACK_SKIP_ROWS,qt),F.pixelStorei(F.UNPACK_SKIP_IMAGES,oi),be===0&&z.generateMipmaps&&F.generateMipmap(St),Oe.unbindTexture()},this.initRenderTarget=function(M){_.get(M).__webglFramebuffer===void 0&&V.setupRenderTarget(M)},this.initTexture=function(M){M.isCubeTexture?V.setTextureCube(M,0):M.isData3DTexture?V.setTexture3D(M,0):M.isDataArrayTexture||M.isCompressedArrayTexture?V.setTexture2DArray(M,0):V.setTexture2D(M,0),Oe.unbindTexture()},this.resetState=function(){B=0,j=0,K=null,Oe.reset(),De.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get coordinateSystem(){return Cn}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(e){this._outputColorSpace=e;const t=this.getContext();t.drawingBufferColorSpace=_t._getDrawingBufferColorSpace(e),t.unpackColorSpace=_t._getUnpackColorSpace()}}const tc=await iu();tc.setup();const{Manifold:cs,Mesh:S_}=tc,co=[new gf({flatShading:!0}),new qo({color:"red",flatShading:!0}),new qo({color:"blue",flatShading:!0})],_r=new In(void 0,co),y_=cs.reserveIDs(co.length),nc=[...Array(co.length)].map((i,e)=>y_+e),ic=new Map;nc.forEach((i,e)=>ic.set(i,e));const uo=new uf,us=new hn(30,1,.01,10);us.position.z=1;us.add(new Ef(16777215,1));uo.add(us);uo.add(_r);const rc=document.querySelector("#output"),$a=new M_({canvas:rc,antialias:!0}),Ml=rc.getBoundingClientRect();$a.setSize(Ml.width,Ml.height);$a.setAnimationLoop(function(i){_r.rotation.x=i/2e3,_r.rotation.y=i/1e3,$a.render(uo,us)});const fs=new Ki(.2,.2,.2);fs.clearGroups();fs.addGroup(0,18,0);fs.addGroup(18,1/0,1);const hs=new lo(.16);hs.clearGroups();hs.addGroup(30,1/0,2);hs.addGroup(0,30,0);const E_=new cs(ac(fs)),b_=new cs(ac(hs));function sc(i){_r.geometry?.dispose(),_r.geometry=T_(cs[i](E_,b_).getMesh())}sc("union");const Sl=document.querySelector("select");Sl.onchange=function(){sc(Sl.value)};function ac(i){const e=i.attributes.position.array,t=i.index!=null?i.index.array:new Uint32Array(e.length/3).map((l,h)=>h),n=[...Array(i.groups.length)].map((l,h)=>i.groups[h].start),r=[...Array(i.groups.length)].map((l,h)=>nc[i.groups[h].materialIndex]),s=Array.from(n.keys());s.sort((l,h)=>n[l]-n[h]);const a=new Uint32Array(s.map(l=>n[l])),o=new Uint32Array(s.map(l=>r[l]));return new S_({numProp:3,vertProperties:e,triVerts:t,runIndex:a,runOriginalID:o}).merge()}function T_(i){const e=new bn;e.setAttribute("position",new _n(i.vertProperties,3)),e.setIndex(new _n(i.triVerts,1));let t=i.runOriginalID[0],n=i.runIndex[0];for(let r=0;r<i.numRun;++r){const s=i.runOriginalID[r+1];if(s!==t){const a=i.runIndex[r+1];e.addGroup(n,a-n,ic.get(t)),t=s,n=a}}return e}
