// B=2 KV8 append/attention replay for speculative verification.
#include <level_zero/ze_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#define CHECK(x) do { ze_result_t r=(x); if(r!=ZE_RESULT_SUCCESS){std::fprintf(stderr,"L0 %d: %s\n",(int)r,#x);std::exit(1);} } while(0)
static constexpr int D=256,NQ=16,NKV=2,GQA=8,TMAX=8,B=2;
struct Ctrl { int token,pos,active,selected; };
static int8_t q8(float x){int q=(int)(x>=0?x+.5f:x-.5f);return(int8_t)(q<-127?-127:(q>127?127:q));}
static float rope(const float*x,int d,int p){if(d>=64)return x[d];int i=d<32?d:d-32;float a=(float)p/std::pow(10000000.f,(float)(2*i)/64.f);return d<32?x[d]*std::cos(a)-x[d+32]*std::sin(a):x[d-32]*std::sin(a)+x[d]*std::cos(a);}
static std::vector<uint8_t> load(const char*p){std::ifstream f(p,std::ios::binary);f.seekg(0,std::ios::end);size_t n=f.tellg();f.seekg(0);std::vector<uint8_t>b(n);f.read((char*)b.data(),n);return b;}

int main(int argc,char**argv){
 if(argc<2)return 2;CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));uint32_t nd=0;CHECK(zeDriverGet(&nd,nullptr));std::vector<ze_driver_handle_t>ds(nd);CHECK(zeDriverGet(&nd,ds.data()));ze_driver_handle_t drv=nullptr;ze_device_handle_t dev=nullptr;
 for(auto d:ds){uint32_t n=0;CHECK(zeDeviceGet(d,&n,nullptr));std::vector<ze_device_handle_t>vs(n);CHECK(zeDeviceGet(d,&n,vs.data()));for(auto v:vs){ze_device_properties_t p{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};CHECK(zeDeviceGetProperties(v,&p));if(p.vendorId==0x8086&&p.deviceId==0x64a0){drv=d;dev=v;}}}if(!dev)return 1;
 ze_context_desc_t cd{ZE_STRUCTURE_TYPE_CONTEXT_DESC,nullptr,0};ze_context_handle_t ctx;CHECK(zeContextCreate(drv,&cd,&ctx));auto spv=load(argv[1]);ze_module_desc_t md{ZE_STRUCTURE_TYPE_MODULE_DESC,nullptr,ZE_MODULE_FORMAT_IL_SPIRV,spv.size(),spv.data(),nullptr,nullptr};ze_module_handle_t mod;CHECK(zeModuleCreate(ctx,dev,&md,&mod,nullptr));
 auto kern=[&](const char*n){ze_kernel_desc_t d{ZE_STRUCTURE_TYPE_KERNEL_DESC,nullptr,0,n};ze_kernel_handle_t k;CHECK(zeKernelCreate(mod,&d,&k));CHECK(zeKernelSetGroupSize(k,256,1,1));return k;};auto ka=kern("kv8_append_batch"),kt=kern("kv8_attn_batch");
 ze_device_mem_alloc_desc_t dd{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,nullptr,0,0};ze_host_mem_alloc_desc_t hd{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,nullptr,0};auto alloc=[&](size_t n,void**p){CHECK(zeMemAllocDevice(ctx,&dd,n,4096,dev,p));};void *dq,*dk,*dv,*dg,*do_,*kc,*vc,*ks,*vs;alloc(B*NQ*D*4,&dq);alloc(B*NKV*D*4,&dk);alloc(B*NKV*D*4,&dv);alloc(B*NQ*D*4,&dg);alloc(B*NQ*D*4,&do_);alloc((size_t)NKV*TMAX*D,&kc);alloc((size_t)NKV*TMAX*D,&vc);alloc(TMAX*NKV*4,&ks);alloc(TMAX*NKV*4,&vs);Ctrl*ctrl;CHECK(zeMemAllocShared(ctx,&dd,&hd,sizeof(Ctrl),64,dev,(void**)&ctrl));
 ze_command_queue_desc_t qd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,nullptr,0,0,0,ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,ZE_COMMAND_QUEUE_PRIORITY_NORMAL};ze_command_list_handle_t up;CHECK(zeCommandListCreateImmediate(ctx,dev,&qd,&up));auto copy=[&](void*d,const void*s,size_t n){CHECK(zeCommandListAppendMemoryCopy(up,d,s,n,nullptr,0,nullptr));};uint8_t z=0;CHECK(zeCommandListAppendMemoryFill(up,kc,&z,1,(size_t)NKV*TMAX*D,nullptr,0,nullptr));CHECK(zeCommandListAppendMemoryFill(up,vc,&z,1,(size_t)NKV*TMAX*D,nullptr,0,nullptr));CHECK(zeCommandListAppendMemoryFill(up,ks,&z,1,TMAX*NKV*4,nullptr,0,nullptr));CHECK(zeCommandListAppendMemoryFill(up,vs,&z,1,TMAX*NKV*4,nullptr,0,nullptr));
 uint64_t seed=0x258ULL;auto rnd=[&](){seed=seed*6364136223846793005ull+1442695040888963407ull;return(float)((int)(seed>>33)-0x3fffffff)/1073741824.f;};std::vector<float>q(B*NQ*D),k(B*NKV*D),v(B*NKV*D),g(B*NQ*D),out(B*NQ*D),out2(B*NQ*D),sc(TMAX*NKV),sv(TMAX*NKV);for(auto&x:q)x=rnd();for(auto&x:k)x=rnd()*3;for(auto&x:v)x=rnd()*4;for(auto&x:g)x=rnd();copy(dq,q.data(),q.size()*4);copy(dk,k.data(),k.size()*4);copy(dv,v.data(),v.size()*4);copy(dg,g.data(),g.size()*4);*ctrl={0,0,2,0};
 auto set=[&](ze_kernel_handle_t h,int i,void*p){CHECK(zeKernelSetArgumentValue(h,i,sizeof(void*),&p));};void*cp=ctrl;set(ka,0,dq);set(ka,1,dk);set(ka,2,dv);set(ka,3,kc);set(ka,4,vc);set(ka,5,ks);set(ka,6,vs);set(ka,7,cp);int tm=TMAX;CHECK(zeKernelSetArgumentValue(ka,8,sizeof(int),&tm));int b=B;CHECK(zeKernelSetArgumentValue(ka,9,sizeof(int),&b));ze_group_count_t ga{B,1,1};CHECK(zeCommandListAppendLaunchKernel(up,ka,&ga,nullptr,0,nullptr));copy(sc.data(),ks,sc.size()*4);copy(sv.data(),vs,sv.size()*4);
 set(kt,0,do_);set(kt,1,dq);set(kt,2,dg);set(kt,3,kc);set(kt,4,vc);set(kt,5,ks);set(kt,6,vs);set(kt,7,cp);CHECK(zeKernelSetArgumentValue(kt,8,sizeof(int),&tm));CHECK(zeKernelSetArgumentValue(kt,9,sizeof(int),&b));ze_group_count_t gt{B*NQ,1,1};CHECK(zeCommandListAppendLaunchKernel(up,kt,&gt,nullptr,0,nullptr));copy(out.data(),do_,out.size()*4);CHECK(zeCommandListAppendLaunchKernel(up,kt,&gt,nullptr,0,nullptr));copy(out2.data(),do_,out2.size()*4);
 double worst=0;bool det=true;for(int bi=0;bi<B;++bi)for(int hq=0;hq<NQ;++hq)for(int d=0;d<D;++d){int h=hq/GQA;float mx=-1e30,sum=0,acc=0;for(int p=0;p<=bi;++p){float dot=0;for(int j=0;j<D;++j){float qq=rope(q.data()+((size_t)bi*NQ+hq)*D,j,bi),kk=rope(k.data()+((size_t)p*NKV+h)*D,j,p);dot+=qq*(q8(kk/sc[p*NKV+h])*sc[p*NKV+h]);}float s=dot/16.f,vv=q8(v[((size_t)p*NKV+h)*D+d]/sv[p*NKV+h])*sv[p*NKV+h];if(s>mx){float e=std::exp(mx-s);acc=acc*e+vv;sum=sum*e+1;mx=s;}else{float e=std::exp(s-mx);acc+=e*vv;sum+=e;}}float want=(acc/sum)/(1+std::exp(-g[((size_t)bi*NQ+hq)*D+d]));worst=std::max(worst,(double)std::fabs(out[((size_t)bi*NQ+hq)*D+d]-want));if(std::memcmp(&out[((size_t)bi*NQ+hq)*D+d],&out2[((size_t)bi*NQ+hq)*D+d],4))det=false;}
 bool ok=det&&worst<1e-3;std::printf("kv8_batch_primitive: %s worst_abs=%.3e deterministic=%d\n",ok?"PASSED":"FAILED",worst,det);if(argc>2){std::ofstream o(argv[2]);o<<"{\"status\":\""<<(ok?"PASSED":"FAILED")<<"\",\"worst_abs\":"<<worst<<",\"bitwise_deterministic\":"<<(det?"true":"false")<<"}\n";}return ok?0:1;
}
