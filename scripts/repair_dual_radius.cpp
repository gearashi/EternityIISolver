// Strict score gains at a specified Hamming distance. Transfer potentials are
// independently verified on all supplied legal placements before pruning.
#include "repair_signed_family.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int max_radius=8;
using Row=std::array<int,max_radius>;
struct Profile {int length=0,best=-100,state[4],delta[4];};
struct Edge {int i,j,s,t,old;};

class Search {
    int size=0, n=0, k=0, gain=0;
    std::uint64_t limit=0,work=0,partial=0,sets=0,bounded=0,alloc=0,tuples=0,setup=0;
    std::vector<std::array<int,4>> faces,neighbors;
    std::vector<std::vector<int>> side;
    std::vector<std::vector<bool>> legal;
    std::vector<bool> anchor,eligible;
    std::vector<int> parent,prices,cost,mutable_cells;
    std::vector<Profile> profiles;
    Row chosen{},solution{};
    const Profile *table[max_radius][max_radius];
    const Profile *assigned[max_radius];
    std::array<Edge,2*max_radius> links;
    int nlinks=0,credit_internal=0;
    int rows_started=0,rows_finished=0;

    void charge(std::uint64_t &counter) {
        if(work>=limit) throw std::overflow_error("work_limit");
        ++work; ++counter;
    }

    void prepare() {
        side.assign(n,std::vector<int>(n,-1));
        anchor.assign(n,false);
        for(int c=0;c<n;c++) for(int s=0;s<4;s++) {
            int d=neighbors[c][s];
            if(d<0) continue;
            side[c][d]=s;
            if(faces[parent[c]][s]!=faces[parent[d]][(s+2)%4]) anchor[c]=true;
        }
        profiles.resize(n*n*16);
        cost.assign(n,100);
        for(int c:mutable_cells) for(int q:mutable_cells) for(int mask=0;mask<16;mask++) {
            auto &p=profiles[(c*n+q)*16+mask];
            for(int s=4*(parent[q]/4);s<4*(parent[q]/4)+4;s++) {
                charge(setup);
                if(!legal[c][s]) continue;
                int ext=0;
                for(int dir=0;dir<4;dir++) {
                    int d=neighbors[c][dir];
                    if(d<0 || (mask&(1<<dir))) continue;
                    int color=faces[parent[d]][(dir+2)%4];
                    ext+=(faces[s][dir]==color)-(faces[parent[c]][dir]==color);
                }
                if(mask==0) {
                    int reduced=prices[q]-prices[c]-ext;
                    if(reduced<0) throw std::invalid_argument("unverified transfer potential");
                    if(s!=parent[c]) cost[c]=std::min(cost[c],reduced);
                }
                if(s==parent[c]) continue;
                p.state[p.length]=s;
                p.delta[p.length++]=ext;
                p.best=std::max(p.best,ext);
            }
        }
    }

    bool orient(int j,int role_delta,Row &states) {
        if(j==k) {
            charge(tuples);
            for(int e=0;e<nlinks;e++) {
                auto l=links[e];
                role_delta+=(faces[states[l.i]][l.s]==faces[states[l.j]][l.t])-l.old;
            }
            if(role_delta<=0) return false;
            gain=role_delta; solution=states; return true;
        }
        auto p=assigned[j];
        for(int i=0;i<p->length;i++) {
            states[j]=p->state[i];
            if(orient(j+1,role_delta+p->delta[i],states)) return true;
        }
        return false;
    }

    bool evaluate() {
        charge(sets);
        Row mask{};
        nlinks=0; credit_internal=0;
        for(int i=0;i<k;i++) for(int j=i+1;j<k;j++) {
            int c=chosen[i],d=chosen[j],s=side[c][d];
            if(s<0) continue;
            int t=(s+2)%4, old=faces[parent[c]][s]==faces[parent[d]][t];
            mask[i]|=1<<s; mask[j]|=1<<t;
            links[nlinks++]={i,j,s,t,old};
            credit_internal+=1-old;
        }
        int relaxed=credit_internal;
        for(int i=0;i<k;i++) {
            int best=-100;
            for(int j=0;j<k;j++) {
                auto p=&profiles[(chosen[i]*n+chosen[j])*16+mask[i]];
                table[i][j]=p; best=std::max(best,p->best);
            }
            relaxed+=best;
        }
        if(relaxed<=0) {bounded++; return false;}
        Row seq{};
        for(int i=0;i<k;i++) seq[i]=i;
        do {
            charge(alloc);
            int upper=credit_internal;
            for(int i=0;i<k;i++) {assigned[i]=table[i][seq[i]]; upper+=assigned[i]->best;}
            Row states{};
            if(upper>0 && orient(0,0,states)) return true;
        } while(std::next_permutation(seq.begin(),seq.begin()+k));
        return false;
    }

    bool extend(int depth,int after,int credit) {
        charge(partial);
        if(depth==k) return credit>0 && evaluate();
        // When current dual credit is nonpositive a final nonneighbor only
        // subtracts its (nonnegative) reduced cost. Try just its grid-neighbors.
        if(depth==k-1 && credit<=0) {
            std::array<int,max_radius*4> tried{};
            int seen=0;
            for(int i=0;i<depth;i++) for(int d:neighbors[chosen[i]]) {
                if(d<after || !eligible[d] || d==chosen[0]) continue;
                if(std::find(tried.begin(),tried.begin()+seen,d)!=tried.begin()+seen) continue;
                tried[seen++]=d;
                if(add(depth,d,credit)) return true;
            }
            return false;
        }
        for(int d=after;d<n;d++) if(eligible[d] && add(depth,d,credit)) return true;
        return false;
    }

    bool add(int depth,int d,int credit) {
        int next=credit-cost[d];
        for(int j=0;j<depth;j++) {
            int c=chosen[j],s=side[c][d];
            if(s>=0) next+=1+(faces[parent[c]][s]==faces[parent[d]][(s+2)%4]);
        }
        chosen[depth]=d;
        return extend(depth+1,d+1,next);
    }

    bool component_family() {
        int count;
        if(!(std::cin>>count) || count<0 || count>400000)
            throw std::invalid_argument("invalid number of connected components");
        for(int row=0;row<count;row++) {
            int len,upper;
            if(!(std::cin>>len>>upper) || len<1 || len>k || upper<1 || upper>20*k)
                throw std::invalid_argument("invalid component bound");
            std::vector<bool> component(n,false);
            for(int i=0;i<len;i++) {
                int c;
                if(!(std::cin>>c) || c<0 || c>=n || component[c] ||
                   std::find(mutable_cells.begin(),mutable_cells.end(),c)==mutable_cells.end())
                    throw std::invalid_argument("ineligible component member");
                component[c]=true;
                chosen[i]=c;
            }
            eligible.assign(n,false);
            for(int c:mutable_cells) {
                if(component[c]) continue;
                bool touches=false;
                for(int d:neighbors[c]) if(d>=0 && component[d]) touches=true;
                eligible[c]=!touches;
            }
            // The caller proved this upper bound on the component's reduced
            // gain. No complementary change touches the component by design.
            // Remaining changes therefore contribute at most their negative
            // reduced costs plus corrections on complementary internal edges.
            ++rows_started;
            if(extend(len,0,upper)) return true;
            ++rows_finished;
        }
        return false;
    }

public:
    bool read() {
        if(!(std::cin>>size>>k>>limit)||size<1||size>16||k<1||k>max_radius||limit==0) return false;
        n=size*size; faces.resize(4*n);parent.resize(n);prices.resize(n);neighbors.resize(n);
        legal.assign(n,std::vector<bool>(4*n));
        for(auto &f:faces) for(int &x:f) if(!(std::cin>>x)) return false;
        for(int &s:parent) if(!(std::cin>>s)||s<0||s>=4*n) return false;
        for(int &p:prices) if(!(std::cin>>p)||p<0||p>4*n) return false;
        for(int c=0;c<n;c++) {
            int m;
            if(!(std::cin>>m)||m<1||m>4*n) return false;
            for(int i=0;i<m;i++) {
                int s;
                if(!(std::cin>>s)||s<0||s>=4*n) return false;
                legal[c][s]=true;
            }
            if(!legal[c][parent[c]]) return false;
            if(m>1) mutable_cells.push_back(c);
            neighbors[c]={c<size?-1:c-size,c%size==size-1?-1:c+1,c+size>=n?-1:c+size,c%size==0?-1:c-1};
        }
        return true;
    }

    void run(bool restricted_family, bool coverage_receipt, bool signed_mode) {
        SignedFamilies signed_catalog;
        bool complete=true;
        try {
            prepare();
            if(signed_mode) {
                signed_catalog.read(k,mutable_cells,neighbors,[&](){ charge(setup); });
                signed_catalog.search([&](){ charge(partial); },
                    [&](const Row &cells){ chosen=cells; return evaluate(); });
            }
            else if(restricted_family) component_family();
            else for(int a:mutable_cells) if(anchor[a]) {
                chosen[0]=a;eligible.assign(n,false);
                for(int d:mutable_cells) eligible[d]=d!=a && (!anchor[d]||d>a);
                if(extend(1,0,-cost[a])) break;
            }
        } catch(const std::overflow_error&) {complete=false;}
        std::cout<<(complete?(restricted_family?"family_complete":"complete"):"work_limit")<<' '<<gain<<' '<<work<<' '<<partial<<' '
                 <<sets<<' '<<bounded<<' '<<alloc<<' '<<tuples<<' '<<setup<<'\n';
        if(gain>0) {
            for(int i=0;i<k;i++) parent[chosen[i]]=solution[i];
            for(int s:parent) std::cout<<s<<' ';
            std::cout<<'\n';
        }
        if(signed_mode) {
            std::cout<<"signed_rows "<<signed_catalog.rows<<' '<<signed_catalog.started<<' '
                     <<signed_catalog.finished<<' '<<signed_catalog.covered<<'\n';
        }
        if(coverage_receipt) {
            std::cout<<"component_rows "<<rows_started<<' '<<rows_finished<<'\n';
        }
    }
};
int main(int argc,char **argv) {
    if(argc>2) return 2;
    const std::string mode=argc==2 ? argv[1] : "";
    const bool receipt=mode=="--component-family-v2";
    const bool signed_mode=mode=="--signed-family";
    if(argc==2 && !receipt && !signed_mode && mode!="--component-family") return 2;
    Search search;
    if(!search.read()) return 2;
    try {search.run(argc==2,receipt,signed_mode);} catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 3;}
}
