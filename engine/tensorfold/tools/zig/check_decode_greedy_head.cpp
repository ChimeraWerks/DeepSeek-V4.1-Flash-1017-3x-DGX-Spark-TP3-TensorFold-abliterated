// CPU only. New production comparator vs independent CURRENT GPU reduction model.
#include "../../zig/kernels/cuda/dsv41_greedy_head.h"
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>
static unsigned long long cases = 0;
static void check(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "FAIL after %llu cases: %s\n", cases, why); std::exit(1); }
}
struct Old { float value; uint32_t id; };
static Old old_pick(Old a, Old b) {
    return b.value > a.value || (b.value == a.value && b.id < a.id) ? b : a;
}
// Faithful thread-strided loop, 32 warp reductions, then shared warp reduction.
// GPU out-of-range shuffle lanes retain their value; they never affect lane 0.
static uint32_t global_gpu(const std::vector<uint32_t>& row, unsigned n) {
    std::array<Old,1024> threads;
    threads.fill({-std::numeric_limits<float>::infinity(), 0x7fffffff});
    for (unsigned lane=0; lane<1024; ++lane)
        for (unsigned i=lane; i<n; i+=1024) {
            const float v=std::bit_cast<float>(row[i]);
            if (v==v) threads[lane]=old_pick(threads[lane],{v,i});
        }
    for (unsigned off=16; off; off>>=1) {
        const auto prev=threads;
        for(unsigned i=0;i<1024;++i)
            threads[i]=old_pick(prev[i],prev[(i%32+off<32)?i+off:i]);
    }
    std::array<Old,32> shared;
    for(unsigned i=0;i<32;++i) shared[i]=threads[i*32];
    for(unsigned off=16;off;off>>=1) {
        const auto prev=shared;
        for(unsigned i=0;i<32;++i) shared[i]=old_pick(prev[i],prev[i+off<32?i+off:i]);
    }
    const float first=std::bit_cast<float>(row[0]);
    return first!=first?0:shared[0].id;
}
static TfDsHeadCandidate local_gpu(const uint32_t* row, unsigned width, unsigned offset, unsigned vocab) {
    std::array<TfDsHeadCandidate,1024> threads;
    threads.fill(tf_ds_head_empty());
    for(unsigned lane=0;lane<1024;++lane)
        for(unsigned i=lane;i<width && offset+i<vocab;i+=1024)
            threads[lane]=tf_ds_head_pick(threads[lane],{row[i],offset+i});
    for(unsigned off=16;off;off>>=1) {
        const auto prev=threads;
        for(unsigned i=0;i<1024;++i)
            threads[i]=tf_ds_head_pick(prev[i],prev[i%32+off<32?i+off:i]);
    }
    std::array<TfDsHeadCandidate,32> shared;
    for(unsigned i=0;i<32;++i) shared[i]=threads[i*32];
    for(unsigned off=16;off;off>>=1) {
        const auto prev=shared;
        for(unsigned i=0;i<32;++i) shared[i]=tf_ds_head_pick(prev[i],prev[i+off<32?i+off:i]);
    }
    return shared[0];
}
static void one(const std::vector<uint32_t>& row,unsigned width,unsigned world,unsigned vocab) {
    ++cases;
    auto best=tf_ds_head_empty();
    for(unsigned rank=0;rank<world;++rank)
        best=tf_ds_head_pick(best,local_gpu(row.data()+rank*width,width,rank*width,vocab));
    const auto id=global_gpu(row,vocab);
    check(best.id==id,"shard merge token differs from current GPU topology");
    check(best.bits==row[id],"winner fp32 bits changed");
}
int main() {
    const uint32_t special[]={0,0x80000000,0x7f800000,0xff800000,0x3f800000,0xbf800000,
        1,0x80000001,0x007fffff,0x807fffff,0x7fc12345,0x7f800001,0xffc54321,0xff800001};
    // Every special-value placement in a three-token row; all ranks can own NaN.
    for(auto a:special)for(auto b:special)for(auto c:special) {
        one({a,b,c},1,3,3);
        one({a,b,c,0x7f800000},2,2,3);
    }
    std::mt19937 rng(0x7dd85b);
    for(unsigned width:{1,2,31,32,33,1023,1024,1025,43136})for(unsigned world:{1,2,3,4}) {
        std::vector<uint32_t> row(width*world);
        for(unsigned trial=0;trial<100;++trial) {
            for(auto& v:row) v=rng();
            const unsigned vocab=trial%3==0?1:trial%3==1?unsigned(row.size()):1+rng()%row.size();
            one(row,width,world,vocab);
        }
        for(auto x:special) {
            std::fill(row.begin(),row.end(),x); one(row,width,world,row.size());
        }
        for(unsigned rank=0;rank<world;++rank) {
            std::fill(row.begin(),row.end(),0xff800000);
            row[rank*width]=0x7fc13579; one(row,width,world,row.size());
            row[rank*width]=0x7f800000; one(row,width,world,row.size());
        }
        if(row.size()>1) {
            std::fill(row.begin(),row.end(),0xbf800000);
            row.front()=0x80000000; row.back()=0; one(row,width,world,row.size());
            row.front()=0; row.back()=0x80000000; one(row,width,world,row.size());
        }
    }
    // Actual TP3 shape, 128 padding entries deliberately toxic.
    std::vector<uint32_t> padded(129408,0xbf800000);
    for(unsigned i=129280;i<padded.size();++i) padded[i]=special[i%14];
    padded[43136]=0x7fc12345; padded[86272]=0x7f800001;
    padded[129279]=0x3f800000; one(padded,43136,3,129280);
    // Rank-major multiple-row layout consumed by the production merge kernel.
    for(unsigned rows:{1,2,16,48}) {
        std::vector<TfDsHeadCandidate> gathered(rows*3);
        for(unsigned r=0;r<rows;++r) for(unsigned rank=0;rank<3;++rank)
            gathered[rank*rows+r]={0x3f800000,unsigned(rank*43136+r)};
        for(unsigned r=0;r<rows;++r) {
            auto best=tf_ds_head_empty();
            for(unsigned rank=0;rank<3;++rank)best=tf_ds_head_pick(best,gathered[rank*rows+r]);
            check(best.id==r,"rank-major row layout");
            ++cases;
        }
    }
    std::printf("PASS decode greedy head: %llu independent GPU-topology/bit/layout cases\n",cases);
}
