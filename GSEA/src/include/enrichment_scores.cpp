#ifndef GSEA_ENRICHMENT_SCORES
#define GSEA_ENRICHMENT_SCORES

#include "math_helpers.cuh"

#ifdef GSEA_OPENMP_ENABLED
#include <omp.h>
#endif

template<
    class exprs_t,
    class posit_t,
    class bitmp_t,
    class enrch_t,
    class index_t,
    bool transposed=false,
    bool kahan=true>
void compute_scores_cpu(
    exprs_t * Correl,    // num_perms x num_genes (constin)
    posit_t * Index,     // num_perms x num_genes (constin)
    bitmp_t * OPath,     // num_genes (constin)
    enrch_t * Score,     // num_paths x num_perms (out)
    index_t num_genes,   // number of genes
    index_t num_perms,   // number of permutations
    index_t num_paths) { // number of pathways


    #ifdef GSEA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (index_t perm = 0; perm < num_perms; perm++) {
        for (index_t path = 0; path < num_paths; path++) {

            enrch_t Bsf = 0;
            enrch_t Res = 0;
            enrch_t Kah = 0;
            enrch_t N_R = 0;
            index_t N_H = 0;

            // first pass: calculate N_R and N_H
           for (index_t gene = 0; gene < num_genes; gene++) {
                index_t index = transposed ? gene*num_perms+perm :
                                             perm*num_genes+gene ;

                const exprs_t value = Correl[index];
                const exprs_t sigma = OPath[Index[index]].getbit(path);
                const enrch_t input = cuabs(value);

                if (kahan) {
                    const enrch_t alpha = sigma ? input - Kah : 0;
                    const enrch_t futre = sigma ? alpha + N_R : N_R;
                    Kah = sigma ? (futre-N_R)-alpha : Kah;
                    N_R = sigma ? futre : N_R;
                } else {
                    N_R = sigma ? N_R + input : N_R;
                }

                N_H += sigma;
           }

           // second pass: calculate ES
           Kah = 0;
           const enrch_t decay = -1.0/(num_genes-N_H);
           for (index_t gene = 0; gene < num_genes; gene++) {
                index_t index = transposed ? gene*num_perms+perm :
                                             perm*num_genes+gene ;

                const exprs_t value = Correl[index];
                const exprs_t sigma = OPath[Index[index]].getbit(path);
                const enrch_t input = sigma ? cuabs(value)/N_R : decay;

                if (kahan) {
                    const enrch_t alpha = input - Kah;
                    const enrch_t futre = alpha + Res;
                    Kah = (futre-Res)-alpha;
                    Res = futre;
                } else {
                    Res += input;
                }

                Bsf  = cuabs(Res) > cuabs(Bsf) ? Res : Bsf;
           }

           // store best enrichment per path
           Score[path*num_perms+perm] = Bsf;
        }
    }
}

#endif
