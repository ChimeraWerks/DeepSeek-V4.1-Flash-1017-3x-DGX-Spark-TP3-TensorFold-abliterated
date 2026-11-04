// Independent full-sort oracle against the production CPU/device merge.
#include "../../zig/kernels/cuda/dsv41_sample_head.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
static unsigned long long cases = 0;
static float value(uint32_t b) { return std::bit_cast<float>(b); }
static bool before(TfDsHeadCandidate a, TfDsHeadCandidate b) {
    return value(a.bits) > value(b.bits) || (value(a.bits) == value(b.bits) && a.id < b.id);
}
static void check(bool yes) { if (!yes) { std::fprintf(stderr,"FAIL sample head case %llu\n",cases); std::abort(); } }
static void one(const std::vector<uint32_t>& values, unsigned width, unsigned vocab, unsigned k, unsigned rows=1) {
    ++cases;
    std::vector<TfDsHeadCandidate> all(3 * rows * (k+1), tf_ds_head_empty());
    std::vector<TfDsHeadCandidate> full;
    bool nan=false;
    for(unsigned i=0;i<vocab;++i) {
        if(std::isnan(value(values[i]))) nan=true;
        else full.push_back({values[i],i});
    }
    std::sort(full.begin(),full.end(),before);
    unsigned flag=nan?1:0;
    if(!nan) {
        check(full.size()>=k);
        unsigned ties=0;
        for(auto x:full) ties += value(x.bits)==value(full[k-1].bits);
        if(ties>2048) flag=2;
    }
    for(unsigned rank=0;rank<3;++rank) {
        std::vector<TfDsHeadCandidate> local;
        unsigned local_nan=0;
        for(unsigned i=rank*width;i<std::min(vocab,(rank+1)*width);++i) {
            if(std::isnan(value(values[i]))) local_nan=1;
            else local.push_back({values[i],i});
        }
        std::sort(local.begin(),local.end(),before);
        unsigned ties=0;
        if(local.size()>=k) for(auto x:local) ties += value(x.bits)==value(local[k-1].bits);
        for(unsigned row=0;row<rows;++row) {
            auto* dst=all.data()+(rank*rows+row)*(k+1);
            std::copy_n(local.begin(),std::min<size_t>(local.size(),k),dst);
            dst[k]={local_nan,ties};
        }
    }
    std::vector<uint32_t> vals(rows*k);
    std::vector<long long> ids(rows*k);
    std::vector<int> flags(rows,-1);
    for(unsigned row=0;row<rows;++row) {
        tf_ds_sample_head_merge(all.data(),row,rows,3,k,vals.data(),ids.data(),flags.data());
        check(flags[row]==int(flag));
        if(flag==0) for(unsigned i=0;i<k;++i) {
            check(vals[row*k+i]==full[i].bits);
            check(ids[row*k+i]==full[i].id);
        }
    }
}
int main() {
    std::mt19937 rng(20261103);
    const uint32_t special[]={0,0x80000000,0x7f800000,0xff800000,0x3f800000,0xbf800000,1,0x80000001};
    for(unsigned width:{1,2,17,64,127,700}) for(unsigned k:{1,20,28,56,64}) {
        if(k>3*width)continue;
        for(unsigned trial=0;trial<100;++trial) {
            std::vector<uint32_t> v(3*width);
            for(auto& x:v) x=special[rng()%8];
            if(trial%11==0)v[rng()%v.size()]=0x7fc12345;
            one(v,width,v.size(),k);
        }
    }
    for(unsigned ties:{2047,2048,2049,3000}) for(unsigned k:{1,20,28,56,64}) {
        std::vector<uint32_t> v(9000,0xbf800000);
        std::fill_n(v.begin(),ties,0x3f800000);
        one(v,3000,9000,k);
    }
    // Huge local ties below a higher global cutoff must not cause fallback.
    std::vector<uint32_t> low(9000,0xbf800000);
    for(unsigned i=0;i<64;++i)low[3000+i]=0x40000000;
    for(unsigned k:{1,20,28,56,64})one(low,3000,9000,k);
    // Production shard offsets and toxic physical padding.
    std::vector<uint32_t> padded(129408,0xbf800000);
    for(unsigned i=0;i<129280;++i)padded[i]=std::bit_cast<uint32_t>(float(int(rng()%1000)-500));
    for(unsigned i=129280;i<padded.size();++i)padded[i]=0x7fc00000;
    for(unsigned rows:{1,2,16,48})for(unsigned k:{1,20,28,56,64})one(padded,43136,129280,k,rows);
    std::printf("PASS sample head: %llu production-merge cases, 0 failures\n",cases);
}
