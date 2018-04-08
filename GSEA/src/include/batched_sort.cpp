#ifndef GSEA_BATCHED_SORT
#define GSEA_BATCHED_SORT


#ifdef UNIT_TEST
#define GSEA_OPENMP_ENABLED
#define CHECK
#endif

#include <iostream>
#include <algorithm>
#include <omp.h>

#ifdef OPENMP_ENABLED
#include <omp.h>
#endif


template<
    class key_t,                       // data type for the keys
    class val_t,                       // data type for the values
    class ind_t,                       // data type for the indices
    bool ascending=false,              // sort order of the primitive
    ind_t chunk_size=1280>              // number of concurrently sorted batches
void pathway_sort_cpu(
    key_t * __restrict__ Keys,         // keys of all batches   (inplace)
    val_t * __restrict__ Source,       // values for one batch  (constin)
    val_t * __restrict__ Target,       // values of all batches (inplace)
    ind_t batch_size,                  // number of entries in a batch
    ind_t num_batches) {               // number of batches

    // get number of chunks
    const ind_t num_chunks = (num_batches+chunk_size-1)/chunk_size;

    #ifdef GSEA_OPENMP_ENABLED
    # pragma omp parallel
    #endif	
    for (ind_t chunk = 0; chunk < num_chunks; chunk++) {

        const ind_t lower = chunk*chunk_size;
        const ind_t upper = std::min(lower+chunk_size, num_batches);

        #ifdef GSEA_OPENMP_ENABLED
        # pragma omp for
        #endif
        for (ind_t batch = lower; batch < upper; batch++) {

            // base index
            const ind_t base = batch*batch_size;

            // auxilliary memory
            std::vector<key_t> locKey(Keys+base, Keys+base+batch_size);
            std::vector<ind_t> locInd(batch_size);
            std::iota(locInd.data(), locInd.data()+batch_size, 0);

            // sort predicate
            auto predicate = [&] (const ind_t& lhs, const ind_t& rhs) {
                if (ascending)
                    return Keys[base+lhs] < Keys[base+rhs];
                else
                    return Keys[base+lhs] > Keys[base+rhs];
            };

            // sort indices
            std::sort(locInd.data(), locInd.data()+batch_size, predicate);

            // substitution
            for (ind_t id = 0; id < batch_size; id++) {
                Keys  [base+id] = locKey[locInd[id]];
                Target[base+id] = Source[locInd[id]];
            }
        }
    }
}

#ifdef UNIT_TEST

uint32_t nvidia_hash(uint32_t x) {

    x = (x + 0x7ed55d16) + (x << 12);
    x = (x ^ 0xc761c23c) ^ (x >> 19);
    x = (x + 0x165667b1) + (x <<  5);
    x = (x + 0xd3a2646c) ^ (x <<  9);
    x = (x + 0xfd7046c5) + (x <<  3);
    x = (x ^ 0xb55a4f09) ^ (x >> 16);

    return x;
}


void check_cpu(){

    typedef int val_t;
    typedef int key_t;
    typedef int ind_t;

    ind_t batch_size = 20000;
    ind_t num_batches = 20000;

    std::cout << batch_size*num_batches*(sizeof(key_t)+sizeof(ind_t))/(1<<20)
              << std::endl;

    val_t * vals;
    val_t * orig;
    key_t * keys;

    mallocHost(&keys, sizeof(key_t)*num_batches*batch_size);         
    mallocHost(&vals, sizeof(val_t)*num_batches*batch_size);         
    mallocHost(&orig, sizeof(val_t)*batch_size);                     

    //TIMERSTART(fill)
    for (ind_t batch = 0; batch < num_batches; batch++)
        for (ind_t elem = 0; elem < batch_size; elem++) {
            keys[batch*batch_size+elem] =  nvidia_hash(elem);
            vals[batch*batch_size+elem] = ~nvidia_hash(elem);
        }

    for (ind_t elem = 0; elem < batch_size; elem++) {
        orig[elem] = elem;
    }
    //TIMERSTOP(fill)


    //TIMERSTART(sort)
    pathway_sort_cpu<key_t, val_t, ind_t, true>(
        keys,
        orig,
        vals,
        num_batches,
        batch_size);                                                     
    //TIMERSTOP(sort)


    #ifdef OUTPUT
    for (ind_t batch = 0; batch < num_batches; batch++)
        for (ind_t elem = 0; elem < batch_size; elem++)
            std::cout << batch << "\t" << elem << "\t"
                      << keys[batch*batch_size+elem] << "\t"
                      << vals[batch*batch_size+elem] << std::endl;
    #endif

    #ifdef CHECK
    //TIMERSTART(check)
    bool check = true;
    for (ind_t batch = 0; batch < num_batches; batch++)
        check &= std::is_sorted(keys+batch*batch_size,
                                keys+(batch+1)*batch_size);
    std::cout << (check ? "sorted" : "not sorted") << std::endl;
    //TIMERSTOP(check)
    #endif
}

int main() {
    //check_gpu();
    check_cpu();
}

#endif

#endif

