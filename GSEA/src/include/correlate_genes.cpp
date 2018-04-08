#ifndef GSEA_CORRELATE_GENES
#define GSEA_CORRELATE_GENES

#include <omp.h>                   // openMP
#include <mpi.h>		   // MPI
#include <assert.h>                // checks
#include "functors.cuh"            // functors
#include "rngpu/rngpu.hpp"         // lightweight curand alternative

#define GSEA_PERMUTATION_SEED (42)

//////////////////////////////////////////////////////////////////////////////
// correlate low-level primitives for CPU and GPU
//////////////////////////////////////////////////////////////////////////////



template <
    class label_t,
    class index_t,
    class value_t,
    class funct_t,
    unsigned int seed=GSEA_PERMUTATION_SEED>
void correlate_mpi(
    value_t * table,       // expression data table (constin)
    label_t * labels,      // class labels for the two phenotypes
    value_t * correl,      // correlation of genes (output)
    index_t num_genes,     // number of genes
    index_t num_type_A,    // number of patients phenotype A
    index_t num_type_B,    // number of patients phenotype B
    index_t num_perms,     // number of permutations
    funct_t accum,         // accumulator functor
    index_t shift=0) {     // permutation shift

    // indices and helper variables
    const index_t lane = num_type_A + num_type_B;

    // for each permutation
    for (index_t perm = 0; perm < num_perms; perm++) {

        // copy label vector
        std::vector<label_t> sigma(lane);
        for (index_t patient = 0; patient < lane; patient++)
            sigma[patient] = labels[patient];

        // create permutation
        if (perm+shift > 0) {
            auto state=get_initial_fast_kiss_state32(perm+shift+seed);
            fisher_yates_shuffle(fast_kiss32, &state, sigma.data(), lane);
        }

        // for each gene accumulate contribution over all patients
        for (index_t gene = 0; gene < num_genes; gene++)
            correl[perm*num_genes+gene] = accum(sigma.data(),
                                                table,
                                                lane,
                                                gene,
                                                num_genes,
                                                num_type_A,
                                                num_type_B);
    }
}


template <
    class value_t,
    class label_t,
    class index_t,
    bool biased=true,         // use biased standard deviation if applicable
    bool fixlow=true,        // adjust low standard deviation if applicable
    bool transposed=false>   // do not change this unless you know better
void correllate_genes_mpi(
    value_t * Exprs,         // expression data  (constin)
    label_t * Labels,        // phenotype labels (constin)
    value_t * Correl,        // correlation of genes (output)
    index_t num_genes,       // number of genes
    index_t num_type_A,      // number patients phenotype A
    index_t num_type_B,      // number patients phenotype B
    index_t num_perms,       // number of permutations
    std::string metric,      // specify the used metric
    index_t shift=0) {       // shift in permutations

    #ifdef GSEA_PRINT_TIMINGS
    //TIMERSTART(host_correlate_genes)
    #endif

    #ifdef GSEA_PRINT_VERBOSE
    std::cout << "STATUS: Correlating " << num_genes << " unique gene symbols "
              << "for " << num_type_A << " patients" << std::endl
              << "STATUS: with phenotype 0 and " << num_type_B
              << " patients with phenotype 1" << std::endl
              << "STATUS: over " << num_perms
              << " permutations (shift=" << shift << ") " << std::endl
              << "STATUS: using " << metric << " as ranking metric on the GPU."
              << std::endl;
    #endif

    // consistency checks
    assert(num_type_A > 0);
    assert(num_type_B > 0);
    assert(num_genes > 0);
    assert(num_perms > 0);

    // choose metric
    if (metric == "naive_diff_of_classes") {
        auto combiner = combine_difference();
        auto functor  = accumulate_naive_mean
                        <decltype(combiner), transposed>(combiner);
        correlate_mpi<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "naive_ratio_of_classes") {
        auto combiner = combine_quotient();
        auto functor  = accumulate_naive_mean
                        <decltype(combiner), transposed>(combiner);
        correlate_mpi<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "naive_log2_ratio_of_classes") {
        auto combiner = combine_log2_quotient();
        auto functor  = accumulate_naive_mean
                        <decltype(combiner), transposed>(combiner);
        correlate_mpi<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "stable_diff_of_classes") {
        auto combiner = combine_difference();
        auto functor  = accumulate_kahan_mean
                        <decltype(combiner), transposed>(combiner);
        correlate_mpi<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "stable_ratio_of_classes") {
        auto combiner = combine_quotient();
        auto functor  = accumulate_kahan_mean
                        <decltype(combiner), transposed>(combiner);
        correlate_mpi<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "stable_log2_ratio_of_classes") {
        auto combiner = combine_log2_quotient();
        auto functor  = accumulate_kahan_mean
                        <decltype(combiner), transposed>(combiner);
        correlate_mpi<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "onepass_signal2noise") {
        auto combiner = combine_signal2noise<fixlow>();
        auto functor  = accumulate_onepass_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_mpi<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "onepass_t_test") {
        auto combiner = combine_T_test<fixlow>();
        auto functor  = accumulate_onepass_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_mpi<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "twopass_signal2noise") {
        auto combiner = combine_signal2noise<fixlow>();
        auto functor  = accumulate_twopass_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_mpi<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "twopass_t_test") {
        auto combiner = combine_T_test<fixlow>();
        auto functor  = accumulate_twopass_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_mpi<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "stable_signal2noise") {
        auto combiner = combine_signal2noise<fixlow>();
        auto functor  = accumulate_knuth_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_mpi<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "stable_t_test") {
        auto combiner = combine_T_test<fixlow>();
        auto functor  = accumulate_knuth_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_mpi<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "overkill_signal2noise") {
        auto combiner = combine_signal2noise<fixlow>();
        auto functor  = accumulate_overkill_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_mpi<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "overkill_t_test") {
        auto combiner = combine_T_test<fixlow>();
        auto functor  = accumulate_overkill_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_mpi<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else {
        std::cout << "ERROR: unknown metric, exiting." << std::endl;
        exit(GSEA_NO_KNOWN_METRIC_SPECIFIED_ERROR);
    }

    #ifdef GSEA_PRINT_TIMINGS
    //TIMERSTOP(host_correlate_genes)
    #endif
}





template <
    class label_t,
    class index_t,
    class value_t,
    class funct_t,
    unsigned int seed=GSEA_PERMUTATION_SEED>
void correlate_cpu(
    value_t * table,       // expression data table (constin)
    label_t * labels,      // class labels for the two phenotypes
    value_t * correl,      // correlation of genes (output)
    index_t num_genes,     // number of genes
    index_t num_type_A,    // number of patients phenotype A
    index_t num_type_B,    // number of patients phenotype B
    index_t num_perms,     // number of permutations
    funct_t accum,         // accumulator functor
    index_t shift=0) {     // permutation shift

    // indices and helper variables
    const index_t lane = num_type_A + num_type_B;

    // for each permutation
    # pragma omp parallel for
    for (index_t perm = 0; perm < num_perms; perm++) {

        // copy label vector
        std::vector<label_t> sigma(lane);
        for (index_t patient = 0; patient < lane; patient++)
            sigma[patient] = labels[patient];

        // create permutation
        if (perm+shift > 0) {
            auto state=get_initial_fast_kiss_state32(perm+shift+seed);
            fisher_yates_shuffle(fast_kiss32, &state, sigma.data(), lane);
        }

        // for each gene accumulate contribution over all patients
        for (index_t gene = 0; gene < num_genes; gene++)
            correl[perm*num_genes+gene] = accum(sigma.data(),
                                                table,
                                                lane,
                                                gene,
                                                num_genes,
                                                num_type_A,
                                                num_type_B);
    }
}

//////////////////////////////////////////////////////////////////////////////
// correlate high-level primitives for CPU and GPU
//////////////////////////////////////////////////////////////////////////////


template <
    class value_t,
    class label_t,
    class index_t,
    bool biased=true,         // use biased standard deviation if applicable
    bool fixlow=true,        // adjust low standard deviation if applicable
    bool transposed=false>   // do not change this unless you know better
void correllate_genes_cpu(
    value_t * Exprs,         // expression data  (constin)
    label_t * Labels,        // phenotype labels (constin)
    value_t * Correl,        // correlation of genes (output)
    index_t num_genes,       // number of genes
    index_t num_type_A,      // number patients phenotype A
    index_t num_type_B,      // number patients phenotype B
    index_t num_perms,       // number of permutations
    std::string metric,      // specify the used metric
    index_t shift=0) {       // shift in permutations

    #ifdef GSEA_PRINT_TIMINGS
    //TIMERSTART(host_correlate_genes)
    #endif

    #ifdef GSEA_PRINT_VERBOSE
    std::cout << "STATUS: Correlating " << num_genes << " unique gene symbols "
              << "for " << num_type_A << " patients" << std::endl
              << "STATUS: with phenotype 0 and " << num_type_B
              << " patients with phenotype 1" << std::endl
              << "STATUS: over " << num_perms
              << " permutations (shift=" << shift << ") " << std::endl
              << "STATUS: using " << metric << " as ranking metric on the GPU."
              << std::endl;
    #endif

    // consistency checks
    assert(num_type_A > 0);
    assert(num_type_B > 0);
    assert(num_genes > 0);
    assert(num_perms > 0);

    // choose metric
    if (metric == "naive_diff_of_classes") {
        auto combiner = combine_difference();
        auto functor  = accumulate_naive_mean
                        <decltype(combiner), transposed>(combiner);
        correlate_cpu<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "naive_ratio_of_classes") {
        auto combiner = combine_quotient();
        auto functor  = accumulate_naive_mean
                        <decltype(combiner), transposed>(combiner);
        correlate_cpu<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "naive_log2_ratio_of_classes") {
        auto combiner = combine_log2_quotient();
        auto functor  = accumulate_naive_mean
                        <decltype(combiner), transposed>(combiner);
        correlate_cpu<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "stable_diff_of_classes") {
        auto combiner = combine_difference();
        auto functor  = accumulate_kahan_mean
                        <decltype(combiner), transposed>(combiner);
        correlate_cpu<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "stable_ratio_of_classes") {
        auto combiner = combine_quotient();
        auto functor  = accumulate_kahan_mean
                        <decltype(combiner), transposed>(combiner);
        correlate_cpu<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "stable_log2_ratio_of_classes") {
        auto combiner = combine_log2_quotient();
        auto functor  = accumulate_kahan_mean
                        <decltype(combiner), transposed>(combiner);
        correlate_cpu<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "onepass_signal2noise") {
        auto combiner = combine_signal2noise<fixlow>();
        auto functor  = accumulate_onepass_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_cpu<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "onepass_t_test") {
        auto combiner = combine_T_test<fixlow>();
        auto functor  = accumulate_onepass_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_cpu<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "twopass_signal2noise") {
        auto combiner = combine_signal2noise<fixlow>();
        auto functor  = accumulate_twopass_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_cpu<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "twopass_t_test") {
        auto combiner = combine_T_test<fixlow>();
        auto functor  = accumulate_twopass_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_cpu<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "stable_signal2noise") {
        auto combiner = combine_signal2noise<fixlow>();
        auto functor  = accumulate_knuth_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_cpu<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "stable_t_test") {
        auto combiner = combine_T_test<fixlow>();
        auto functor  = accumulate_knuth_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_cpu<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "overkill_signal2noise") {
        auto combiner = combine_signal2noise<fixlow>();
        auto functor  = accumulate_overkill_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_cpu<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else if (metric == "overkill_t_test") {
        auto combiner = combine_T_test<fixlow>();
        auto functor  = accumulate_overkill_stats
                        <decltype(combiner), transposed>(combiner);
        correlate_cpu<label_t>
                     (Exprs, Labels, Correl, num_genes, num_type_A,
                      num_type_B, num_perms, functor, shift);             

    } else {
        std::cout << "ERROR: unknown metric, exiting." << std::endl;
        exit(GSEA_NO_KNOWN_METRIC_SPECIFIED_ERROR);
    }

    #ifdef GSEA_PRINT_TIMINGS
    //TIMERSTOP(host_correlate_genes)
    #endif
}

#endif
