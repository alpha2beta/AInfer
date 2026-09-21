#include <level_zero/ze_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <vector>

#define CHECK(x) do { ze_result_t r=(x); if(r!=ZE_RESULT_SUCCESS){std::fprintf(stderr,"L0 %d: %s\n",(int)r,#x);std::exit(1);} } while(0)
static constexpr int D=256, NQ=16, NKV=2, GQA=8, T=37;
struct Ctrl { int token, pos, active, selected; };
static int8_t q8(float x) { int q=(int)(x>=0?x+.5f:x-.5f); return (int8_t)std::max(-127,std::min(127,q)); }
static float rope(const float *x,int d,int pos) {
  if(d>=64)return x[d]; int i=d<32?d:d-32; float a=(float)pos/std::pow(10000000.f,(float)(2*i)/64.f);
  return d<32?x[d]*std::cos(a)-x[d+32]*std::sin(a):x[d-32]*std::sin(a)+x[d]*std::cos(a);
}
static std::vector<uint8_t> bytes(const char *p) { std::ifstream f(p,std::ios::binary); f.seekg(0,std::ios::end); size_t n=f.tellg(); f.seekg(0); std::vector<uint8_t>b(n);f.read((char*)b.data(),n);return b; }

int main(int argc,char **argv) {
  if(argc<2){std::fprintf(stderr,"usage: %s kv8_primitive.spv [report.json]\n",argv[0]);return 2;}
  CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY)); uint32_t nd=0; CHECK(zeDriverGet(&nd,nullptr)); std::vector<ze_driver_handle_t> ds(nd); CHECK(zeDriverGet(&nd,ds.data()));
  ze_driver_handle_t drv=nullptr; ze_device_handle_t dev=nullptr;
  for(auto d:ds){uint32_t n=0;CHECK(zeDeviceGet(d,&n,nullptr));std::vector<ze_device_handle_t>vs(n);CHECK(zeDeviceGet(d,&n,vs.data()));for(auto v:vs){ze_device_properties_t p{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};CHECK(zeDeviceGetProperties(v,&p));if(p.vendorId==0x8086&&p.deviceId==0x64a0){drv=d;dev=v;}}}
  if(!dev)return 1; ze_context_desc_t cd{ZE_STRUCTURE_TYPE_CONTEXT_DESC,nullptr,0};ze_context_handle_t ctx;CHECK(zeContextCreate(drv,&cd,&ctx));
  auto spv=bytes(argv[1]); ze_module_desc_t md{ZE_STRUCTURE_TYPE_MODULE_DESC,nullptr,ZE_MODULE_FORMAT_IL_SPIRV,spv.size(),spv.data(),nullptr,nullptr};ze_module_handle_t mod;CHECK(zeModuleCreate(ctx,dev,&md,&mod,nullptr));
  auto kernel=[&](const char*n){ze_kernel_desc_t d{ZE_STRUCTURE_TYPE_KERNEL_DESC,nullptr,0,n};ze_kernel_handle_t k;CHECK(zeKernelCreate(mod,&d,&k));return k;};
  ze_kernel_handle_t ka=kernel("kv8_append_ctrl"), kt=kernel("kv8_attn_ctrl"); CHECK(zeKernelSetGroupSize(ka,256,1,1));CHECK(zeKernelSetGroupSize(kt,256,1,1));
  ze_device_mem_alloc_desc_t dd{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,nullptr,0,0};ze_host_mem_alloc_desc_t hd{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,nullptr,0};
  auto alloc=[&](size_t n,void **p){CHECK(zeMemAllocDevice(ctx,&dd,n,4096,dev,p));};
  void *dq,*dk,*dv,*dg,*do_,*kc,*vc,*ks,*vs; alloc(NQ*D*4,&dq);alloc(NKV*D*4,&dk);alloc(NKV*D*4,&dv);alloc(NQ*D*4,&dg);alloc(NQ*D*4,&do_);alloc((size_t)NKV*T*D,&kc);alloc((size_t)NKV*T*D,&vc);alloc((size_t)T*NKV*4,&ks);alloc((size_t)T*NKV*4,&vs);
  Ctrl *ctrl=nullptr;CHECK(zeMemAllocShared(ctx,&dd,&hd,sizeof(Ctrl),64,dev,(void**)&ctrl));
  ze_command_queue_desc_t qd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,nullptr,0,0,0,ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,ZE_COMMAND_QUEUE_PRIORITY_NORMAL};ze_command_queue_handle_t q;CHECK(zeCommandQueueCreate(ctx,dev,&qd,&q));ze_command_list_handle_t up;CHECK(zeCommandListCreateImmediate(ctx,dev,&qd,&up));
  auto copy=[&](void*d,const void*s,size_t n){CHECK(zeCommandListAppendMemoryCopy(up,d,s,n,nullptr,0,nullptr));};
  uint8_t z=0;CHECK(zeCommandListAppendMemoryFill(up,kc,&z,1,(size_t)NKV*T*D,nullptr,0,nullptr));CHECK(zeCommandListAppendMemoryFill(up,vc,&z,1,(size_t)NKV*T*D,nullptr,0,nullptr));CHECK(zeCommandListAppendMemoryFill(up,ks,&z,1,(size_t)T*NKV*4,nullptr,0,nullptr));CHECK(zeCommandListAppendMemoryFill(up,vs,&z,1,(size_t)T*NKV*4,nullptr,0,nullptr));
  uint64_t seed=0x258ULL; // deterministic, arbitrary seed
  auto rnd=[&](){seed=seed*6364136223846793005ull+1442695040888963407ull;return (float)((int)(seed>>33)-0x3fffffff)*(1.f/1073741824.f);};
  std::vector<float> hq(NQ*D),hg(NQ*D),hk(NKV*D),hv(NKV*D),hout(NQ*D),hout2(NQ*D),rk((size_t)T*NKV*D),rv((size_t)T*NKV*D),rs((size_t)T*NKV*2);
  for(auto&x:hq)x=rnd();for(auto&x:hg)x=rnd();copy(dg,hg.data(),hg.size()*4);
  auto set=[&](ze_kernel_handle_t k,int i,void*p){CHECK(zeKernelSetArgumentValue(k,i,sizeof(void*),&p));};
  set(ka,0,dq);set(ka,1,dk);set(ka,2,dv);set(ka,3,kc);set(ka,4,vc);set(ka,5,ks);set(ka,6,vs);void *cp=ctrl;set(ka,7,cp);int tmax=T;CHECK(zeKernelSetArgumentValue(ka,8,sizeof(int),&tmax));
  ze_group_count_t ga{1,1,1};
  for(int pos=0;pos<T;++pos){for(auto&x:hk)x=rnd()*3;for(auto&x:hv)x=rnd()*4;copy(dq,hq.data(),hq.size()*4);copy(dk,hk.data(),hk.size()*4);copy(dv,hv.data(),hv.size()*4);*ctrl={0,pos,pos+1,0};CHECK(zeCommandListAppendLaunchKernel(up,ka,&ga,nullptr,0,nullptr));for(int h=0;h<NKV;++h){float mk=0,mv=0;for(int d=0;d<D;++d){float k=rope(hk.data()+h*D,d,pos),v=hv[h*D+d];mk=std::max(mk,std::fabs(k));mv=std::max(mv,std::fabs(v));}float sk=mk?mk/127:1,sv=mv?mv/127:1;rs[(pos*NKV+h)*2]=sk;rs[(pos*NKV+h)*2+1]=sv;for(int d=0;d<D;++d){rk[((size_t)pos*NKV+h)*D+d]=rope(hk.data()+h*D,d,pos);rv[((size_t)pos*NKV+h)*D+d]=hv[h*D+d];}}}
  std::vector<int8_t> got((size_t)NKV*T*D);std::vector<float> gks(T*NKV),gvs(T*NKV);copy(got.data(),kc,got.size());copy(gks.data(),ks,gks.size()*4);copy(gvs.data(),vs,gvs.size()*4);
  bool slots_ok=true, deterministic=true;int max_q_delta=0;double worst=0,scale_rel=0;for(int p=0;p<T;++p)for(int h=0;h<NKV;++h){float wantk=rs[(p*NKV+h)*2],wantv=rs[(p*NKV+h)*2+1];scale_rel=std::max(scale_rel,(double)std::fabs(gks[p*NKV+h]-wantk)/wantk);scale_rel=std::max(scale_rel,(double)std::fabs(gvs[p*NKV+h]-wantv)/wantv);for(int d=0;d<D;++d){int want=q8(rk[((size_t)p*NKV+h)*D+d]/gks[p*NKV+h]);int delta=std::abs((int)got[((size_t)h*T+p)*D+d]-want);max_q_delta=std::max(max_q_delta,delta);if(delta>1){if(slots_ok)std::fprintf(stderr,"slot mismatch p%d h%d d%d got %d want %d\n",p,h,d,(int)got[((size_t)h*T+p)*D+d],want);slots_ok=false;}}}
  // The final append left dQ RoPE-transformed for position T-1; do not
  // overwrite it with the unrotated host input before testing attention.
  *ctrl={0,T-1,T,0};set(kt,0,do_);set(kt,1,dq);set(kt,2,dg);set(kt,3,kc);set(kt,4,vc);set(kt,5,ks);set(kt,6,vs);set(kt,7,cp);CHECK(zeKernelSetArgumentValue(kt,8,sizeof(int),&tmax));ze_group_count_t gt{NQ,1,1};CHECK(zeCommandListAppendLaunchKernel(up,kt,&gt,nullptr,0,nullptr));copy(hout.data(),do_,hout.size()*4);CHECK(zeCommandListAppendLaunchKernel(up,kt,&gt,nullptr,0,nullptr));copy(hout2.data(),do_,hout2.size()*4);
  for(int qh=0;qh<NQ;++qh)for(int d=0;d<D;++d){float mx=-1e30,sum=0,acc=0;for(int p=0;p<T;++p){int h=qh/GQA;float dot=0;for(int j=0;j<D;++j)dot+=rope(hq.data()+qh*D,j,T-1)*(q8(rk[((size_t)p*NKV+h)*D+j]/gks[p*NKV+h])*gks[p*NKV+h]);float sc=dot/16;float v=q8(rv[((size_t)p*NKV+h)*D+d]/gvs[p*NKV+h])*gvs[p*NKV+h];if(sc>mx){float e=std::exp(mx-sc);acc=acc*e+v;sum=sum*e+1;mx=sc;}else{float e=std::exp(sc-mx);acc+=e*v;sum+=e;}}float want=(acc/sum)/(1+std::exp(-hg[qh*D+d]));worst=std::max(worst,(double)std::fabs(hout[qh*D+d]-want));if(std::memcmp(&hout[qh*D+d],&hout2[qh*D+d],4))deterministic=false;}
  bool ok=slots_ok&&deterministic&&scale_rel<5e-7&&worst<1e-3;
  std::printf("kv8_primitive: %s worst_abs=%.3e scale_rel=%.3e max_q_delta=%d slots=%d deterministic=%d\n",ok?"PASSED":"FAILED",worst,scale_rel,max_q_delta,slots_ok,deterministic);
  if(argc>2){std::ofstream o(argv[2]);o<<"{\"status\":\""<<(ok?"PASSED":"FAILED")<<"\",\"tmax\":"<<T<<",\"worst_abs\":"<<worst<<",\"scale_rel\":"<<scale_rel<<",\"max_q_delta\":"<<max_q_delta<<",\"bounded_slots\":"<<(slots_ok?"true":"false")<<",\"bitwise_deterministic\":"<<(deterministic?"true":"false")<<"}\n";}
  return ok?0:1;
}
