#ifndef GSEA_DUMP_BINARY
#define GSEA_DUMP_BINARY

#include <string>
#include <fstream>

template <class index_t, class value_t>
void dump_binary(const value_t * data, const index_t L, std::string filename) {

    #ifdef GSEA_PRINT_TIMINGS
    //TIMERSTART(host_dump_to_disk)
    #endif

    #ifdef GSEA_PRINT_VERBOSE
    std::cout << "STATUS: Dumping " << (sizeof(value_t)*L) << " many bytes to "
              << filename << "." << std::endl;
    #endif

    // write out file for visual inspection
    std::ofstream ofile(filename.c_str(), std::ios::binary);
    ofile.write((char*) data, sizeof(value_t)*L);
    ofile.close();

    #ifdef GSEA_PRINT_TIMINGS
    //TIMERSTOP(host_dump_to_disk)
    #endif
}

#endif
