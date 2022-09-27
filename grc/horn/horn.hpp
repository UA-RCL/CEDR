#ifndef HORN_HPP
#define HORN_HPP
/********************
GNU Radio C++ Flow Graph Header File

Title: horn
GNU Radio version: 3.8.2.0
********************/

/********************
** Create includes
********************/
#include <gnuradio/top_block.h>
#include <gnuradio/blocks/vector_sink.h>
#include <gnuradio/blocks/vector_source.h>
#include <dash/fft.h>


using namespace gr;



class horn {

private:
    blocks::vector_source<gr_complex>::sptr blocks_vector_source_0;
    blocks::vector_source<gr_complex>::sptr blocks_vector_source_1;
    dash::fft::sptr blocks_fft_0;
    dash::fft::sptr blocks_fft_1;

// Variables:
    int samp_rate = 32000;

public:
    top_block_sptr tb;
    blocks::vector_sink<gr_complex>::sptr blocks_vector_sink_0;
    blocks::vector_sink<gr_complex>::sptr blocks_vector_sink_1;
    horn();
    ~horn();

    int get_samp_rate () const;
    void set_samp_rate(int samp_rate);

};


#endif

