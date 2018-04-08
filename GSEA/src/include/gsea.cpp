#ifndef GSEA_CORE
#define GSEA_CORE


#include <iostream>                         // std::cout
#include <vector>                           // std::vector
#include <iomanip>                          // std::setprecision
#include <numeric>

#include "error_codes.cpp"                  // error codes			
#include "bitmap_types.cpp"                 // bitmap types for pathways
#include "enrichment_scores.cpp"            // compute enrichments
#include "correlate_genes.cpp"              // correlate genes
#include "read_data_files.cpp"              // read gmt, gct and cls
#include "batch_scheduler.cpp"              // compute batches if ram too small
#include "create_bitmaps.cpp"               // pathway representation
#include "batched_sort.cpp"                 // sort a bunch of arrays
#include "dump_binary.cpp"                  // write result to disk
#include "statistics.cpp"                   // final statistics
#include "print_info.cpp"                   // information of the gsea routine



template <
    class exprs_t,                          // data type for expression data
    class index_t,                          // data type for indexing
    class label_t,                          // data type for storing labels
    class bitmp_t,                          // data type for storing pathways
    class enrch_t>                          // data type for enrichment scores
void compute_gsea_cpu(
    std::string gct_filename,               // name of the expression data file
    std::string cls_filename,               // name of the class label file
    std::string gmt_filename,               // name of the pathway data file
    std::string metric,                     // ranking metric specifier
    index_t num_perms,                      // number of permutations (incl. e)
    bool sort_direction=true,               // 1: descending, 0: ascending
    bool swap_labels=false,                 // 1: swap phenotypes, 0: as is
    std::string dump_filename="") {

    // make sure the user does not use bool as label_t to avoid problems
   static_assert(!std::is_same<label_t, bool>::value,
                  "You can't use bool for label_t! Use unsigned char instead.");

    #ifdef GSEA_PRINT_TIMINGS
//    TIMERSTART(host_overall)
    #endif

    #ifdef GSEA_PRINT_WARNINGS
    if (sizeof(label_t) > 1)
        std::cout << "WARNING: Choose a narrower label_t, e.g. char."
                  << std::endl;
    if (!sort_direction)
        std::cout << "WARNING: You chose a non default sort direction."
                  << std::endl;
    if (swap_labels)
        std::cout << "WARNING: You interchanged the roles of the phenotypes."
                  << std::endl;
    #endif

    // used variables
    index_t num_paths = 0;                  // number of pathways
    index_t num_genes = 0;                  // number of unique gene symbols
    index_t num_type_A = 0;                 // number of patients class 0
    index_t num_type_B = 0;                 // number of patients class 1
    index_t num_patients = 0;               // number of overall patients

    std::vector<bitmp_t> opath;             // bitmap for each gene
    std::vector<exprs_t> exprs;             // expression data
    std::vector<label_t> labels;            // class label distribution
    std::vector<std::string> gsymb;         // list of gene symbol name
    std::vector<std::string> plist;         // list of patient names
    std::vector<std::string> pname;         // list of pathway names
    std::vector<std::vector<std::string>> pathw;

    // read expression data from gct file
    read_gct(gct_filename, exprs, plist, gsymb, num_genes, num_patients);

	

    #ifdef GSEA_PRINT_TIMINGS
    //TIMERSTART(host_exclusive_gct_parsing)
    #endif

    // read class label from cls file
    read_cls(cls_filename, labels, num_type_A, num_type_B);

    if (num_patients != num_type_A + num_type_B) {
        std::cout << "ERROR: Incompatible class file (cls) and expression "
                  << "data file, exiting." << std::endl;
        exit(GSEA_INCOMPATIBLE_GCT_AND_CLS_FILE_ERROR);
    }

    // swap labels if demanded by user
    if (swap_labels) {
        for (auto& label : labels)
            label = (label+1) & 1;
        std::swap(num_type_A, num_type_B);
    }

    // read pathways from gmt file
    read_gmt(gmt_filename, pname, pathw, num_paths);


    #ifdef GSEA_PRINT_INFO
    print_gsea_info(num_patients, num_type_A, num_type_B, num_genes,
                    num_paths, num_perms, metric, gct_filename, cls_filename,
                    gmt_filename, sort_direction, swap_labels);
    #endif

    // create indices for genes
    std::vector<unsigned int> oindx(num_genes);
    std::iota(oindx.begin(), oindx.end(), 0);



    // schedule computation
    index_t batch_size_perms = 0, num_batches_perms = 0,
            batch_size_paths = 0, num_batches_paths = 0, num_free_bytes = 0;
    schedule_computation_cpu<exprs_t, bitmp_t, enrch_t>
                            (num_genes, num_perms, num_paths, batch_size_perms,
                             num_batches_perms, batch_size_paths,
                             num_batches_paths, num_free_bytes);




    #ifdef GSEA_PRINT_INFO
    print_scheduler_info(batch_size_perms, num_batches_perms,
                         batch_size_paths, num_batches_paths);
    #endif



    // vector for the storage of the enrichment scores
    std::vector<enrch_t> global_result(num_paths*num_perms);

    // for each batch of perms
   for (index_t batch_pi = 0; batch_pi < num_batches_perms; batch_pi++) {
        const index_t lower_pi = batch_pi*batch_size_perms;
        const index_t upper_pi = cumin(lower_pi+batch_size_perms,num_perms);
        const index_t width_pi = upper_pi-lower_pi;

        // initialize space for the correlation matrix and correlate
        std::vector<exprs_t> correl(width_pi*num_genes, 0);
        correllate_genes_cpu(exprs.data(), labels.data(), correl.data(),
                             num_genes, num_type_A, num_type_B,
                             width_pi, metric, lower_pi);


        std::vector<unsigned int>index(num_genes*width_pi);


        if (sort_direction)
            pathway_sort_cpu<exprs_t, unsigned int, index_t, false>
            (correl.data(), oindx.data(), index.data(), num_genes, width_pi);
        else
            pathway_sort_cpu<exprs_t, unsigned int, index_t, true>
            (correl.data(), oindx.data(), index.data(), num_genes, width_pi);

        // for each batch of paths
        for (index_t batch_pa = 0; batch_pa < num_batches_paths; batch_pa++) {
            const index_t lower_pa = batch_pa*batch_size_paths;
            const index_t upper_pa = cumin(lower_pa+batch_size_paths,num_paths);
            const index_t width_pa = upper_pa-lower_pa;

            // create bitmaps for pathways representation:
            create_bitmaps(gsymb, pathw, opath, lower_pa, width_pa);

            #ifdef GSEA_PRINT_INFO
            print_batch_info(lower_pa, upper_pa, width_pa,
                             lower_pi, upper_pi, width_pi);
            #endif

            // compute enrichment scores
            std::vector<enrch_t> local_result(width_pa*width_pi, 0);
            compute_scores_cpu(correl.data(),
                               index.data(),
                               opath.data(),
                               local_result.data(),
                               num_genes,
                               width_pi,
                               width_pa);

            // strided copy
            for (index_t pa_ind = 0; pa_ind < width_pa; pa_ind++) {
                const index_t base_pa = (lower_pa+pa_ind)*num_perms;
                for (index_t pi_ind = 0; pi_ind < width_pi; pi_ind++) {
                    global_result[base_pa+lower_pi+pi_ind]
                    =local_result[pa_ind*width_pi+pi_ind];
                }
            }
        }


    }

    // dump data to disk if demanded
    if (dump_filename.size()) {
        dump_filename += "_"+std::to_string(num_paths);
        dump_filename += "_"+std::to_string(num_perms);
        dump_filename += "_"+std::to_string(8*sizeof(enrch_t));
        dump_filename += ".es";
        dump_binary(global_result.data(), num_paths*num_perms, dump_filename);
    }

    // write the result to command line
    auto result = final_statistics(global_result, num_paths, num_perms);
    for (index_t path = 0; path < num_paths; path++) {
        std::cout << std::showpos << std::fixed << "RESULT:"
                  << " ES: "   << result[path]
                  << " NES: "  << result[1*num_paths+path]
                  << " NP: "   << result[2*num_paths+path]
                  << " FWER: " << result[3*num_paths+path]
                  << "\t(" << pname[path] << ")" << std::endl;
    }

    #ifdef GSEA_PRINT_TIMINGS
    //TIMERSTOP(host_overall_exclusive_gct_parsing)
    #endif

    #ifdef GSEA_PRINT_TIMINGS
    //TIMERSTOP(host_overall)
    #endif


}

//********************************

template <
    class exprs_t,                          // data type for expression data
    class index_t,                          // data type for indexing
    class label_t,                          // data type for storing labels
    class bitmp_t,                          // data type for storing pathways
    class enrch_t>                          // data type for enrichment scores
void compute_gsea_mpi(
    std::string gct_filename,               // name of the expression data file
    std::string cls_filename,               // name of the class label file
    std::string gmt_filename,               // name of the pathway data file
    std::string metric,                     // ranking metric specifier
    index_t num_perms,                      // number of permutations (incl. e)
    bool sort_direction=true,               // 1: descending, 0: ascending
    bool swap_labels=false,                 // 1: swap phenotypes, 0: as is
    std::string dump_filename="",
    int numP = 4,
    int id = 0) {

	printf("numero de procesos: %d", numP);
	printf("id: %d", id); 

	if (id == 0){

    // make sure the user does not use bool as label_t to avoid problems
   static_assert(!std::is_same<label_t, bool>::value,
                  "You can't use bool for label_t! Use unsigned char instead.");

    #ifdef GSEA_PRINT_TIMINGS
//    TIMERSTART(host_overall)
    #endif

    #ifdef GSEA_PRINT_WARNINGS
    if (sizeof(label_t) > 1)
        std::cout << "WARNING: Choose a narrower label_t, e.g. char."
                  << std::endl;
    if (!sort_direction)
        std::cout << "WARNING: You chose a non default sort direction."
                  << std::endl;
    if (swap_labels)
        std::cout << "WARNING: You interchanged the roles of the phenotypes."
                  << std::endl;
    #endif

    // used variables
    index_t num_paths = 0;                  // number of pathways
    index_t num_genes = 0;                  // number of unique gene symbols
    index_t num_type_A = 0;                 // number of patients class 0
    index_t num_type_B = 0;                 // number of patients class 1
    index_t num_patients = 0;               // number of overall patients

    std::vector<bitmp_t> opath;             // bitmap for each gene
    std::vector<exprs_t> exprs;             // expression data
    std::vector<label_t> labels;            // class label distribution
    std::vector<std::string> gsymb;         // list of gene symbol name
    std::vector<std::string> plist;         // list of patient names
    std::vector<std::string> pname;         // list of pathway names
    std::vector<std::vector<std::string>> pathw;

    // read expression data from gct file
    read_gct(gct_filename, exprs, plist, gsymb, num_genes, num_patients);

	

    #ifdef GSEA_PRINT_TIMINGS
    //TIMERSTART(host_exclusive_gct_parsing)
    #endif

    // read class label from cls file
    read_cls(cls_filename, labels, num_type_A, num_type_B);

    if (num_patients != num_type_A + num_type_B) {
        std::cout << "ERROR: Incompatible class file (cls) and expression "
                  << "data file, exiting." << std::endl;
        exit(GSEA_INCOMPATIBLE_GCT_AND_CLS_FILE_ERROR);
    }

    // swap labels if demanded by user
    if (swap_labels) {
        for (auto& label : labels)
            label = (label+1) & 1;
        std::swap(num_type_A, num_type_B);
    }

    // read pathways from gmt file
    read_gmt(gmt_filename, pname, pathw, num_paths);


    #ifdef GSEA_PRINT_INFO
    print_gsea_info(num_patients, num_type_A, num_type_B, num_genes,
                    num_paths, num_perms, metric, gct_filename, cls_filename,
                    gmt_filename, sort_direction, swap_labels);
    #endif

    // create indices for genes
    std::vector<unsigned int> oindx(num_genes);
    std::iota(oindx.begin(), oindx.end(), 0);



    // schedule computation
    index_t batch_size_perms = 0, num_batches_perms = 0,
            batch_size_paths = 0, num_batches_paths = 0, num_free_bytes = 0;
    schedule_computation_cpu<exprs_t, bitmp_t, enrch_t>
                            (num_genes, num_perms, num_paths, batch_size_perms,
                             num_batches_perms, batch_size_paths,
                             num_batches_paths, num_free_bytes);




    #ifdef GSEA_PRINT_INFO
    print_scheduler_info(batch_size_perms, num_batches_perms,
                         batch_size_paths, num_batches_paths);
    #endif



    // vector for the storage of the enrichment scores
    std::vector<enrch_t> global_result(num_paths*num_perms);

	}


    // for each batch of perms
   for (index_t batch_pi = 0; batch_pi < num_batches_perms; batch_pi++) {

        const index_t lower_pi = batch_pi*batch_size_perms;
        const index_t upper_pi = cumin(lower_pi+batch_size_perms,num_perms);
        const index_t width_pi = upper_pi-lower_pi;

        // initialize space for the correlation matrix and correlate
        std::vector<exprs_t> correl(width_pi*num_genes, 0);
       correllate_genes_mpi(exprs.data(), labels.data(), correl.data(), num_genes, 			num_type_A, num_type_B, width_pi, metric, lower_pi);

        std::vector<unsigned int>index(num_genes*width_pi);


        if (sort_direction)
            pathway_sort_cpu<exprs_t, unsigned int, index_t, false>
            (correl.data(), oindx.data(), index.data(), num_genes, width_pi);
        else
            pathway_sort_cpu<exprs_t, unsigned int, index_t, true>
            (correl.data(), oindx.data(), index.data(), num_genes, width_pi);

        // for each batch of paths
        for (index_t batch_pa = 0; batch_pa < num_batches_paths; batch_pa++) {
            const index_t lower_pa = batch_pa*batch_size_paths;
            const index_t upper_pa = cumin(lower_pa+batch_size_paths,num_paths);
            const index_t width_pa = upper_pa-lower_pa;

            // create bitmaps for pathways representation:
            create_bitmaps(gsymb, pathw, opath, lower_pa, width_pa);

            #ifdef GSEA_PRINT_INFO
            print_batch_info(lower_pa, upper_pa, width_pa,
                             lower_pi, upper_pi, width_pi);
            #endif

            // compute enrichment scores
            std::vector<enrch_t> local_result(width_pa*width_pi, 0);
            compute_scores_cpu(correl.data(),
                               index.data(),
                               opath.data(),
                               local_result.data(),
                               num_genes,
                               width_pi,
                               width_pa);

            // strided copy
            for (index_t pa_ind = 0; pa_ind < width_pa; pa_ind++) {
                const index_t base_pa = (lower_pa+pa_ind)*num_perms;
                for (index_t pi_ind = 0; pi_ind < width_pi; pi_ind++) {
                    global_result[base_pa+lower_pi+pi_ind]
                    =local_result[pa_ind*width_pi+pi_ind];
                }
            }
        }


    }

    // dump data to disk if demanded
    if (dump_filename.size()) {
        dump_filename += "_"+std::to_string(num_paths);
        dump_filename += "_"+std::to_string(num_perms);
        dump_filename += "_"+std::to_string(8*sizeof(enrch_t));
        dump_filename += ".es";
        dump_binary(global_result.data(), num_paths*num_perms, dump_filename);
    }

    // write the result to command line
    auto result = final_statistics(global_result, num_paths, num_perms);
    for (index_t path = 0; path < num_paths; path++) {
        std::cout << std::showpos << std::fixed << "RESULT:"
                  << " ES: "   << result[path]
                  << " NES: "  << result[1*num_paths+path]
                  << " NP: "   << result[2*num_paths+path]
                  << " FWER: " << result[3*num_paths+path]
                  << "\t(" << pname[path] << ")" << std::endl;
    }

    #ifdef GSEA_PRINT_TIMINGS
    //TIMERSTOP(host_overall_exclusive_gct_parsing)
    #endif

    #ifdef GSEA_PRINT_TIMINGS
    //TIMERSTOP(host_overall)
    #endif


}


#endif