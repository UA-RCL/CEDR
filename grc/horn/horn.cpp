/********************
GNU Radio C++ Flow Graph Source File

Title: horn
GNU Radio version: 3.8.2.0
********************/

#include "horn.hpp"
using namespace gr;

const std::vector<gr_complex> test_vec{{1,0},{0,0},{0,0},{0,0}};
const std::vector<gr_complex> test_vec2{{1,0},{1,0},{1,0},{1,0},{1,0},{1,0},{1,0},{1,0}};
const std::vector<tag_t> tags;

horn::horn () {



    this->tb = gr::make_top_block("horn");

// Blocks:
    {
        this->blocks_vector_source_0 = blocks::vector_source<gr_complex>::make(test_vec, false, (unsigned int) 4, tags);
    }
    {
        this->blocks_vector_source_1 = blocks::vector_source<gr_complex>::make(test_vec2, false, (unsigned int) 8, tags);
    }
    {
        this->blocks_vector_sink_0 = blocks::vector_sink<gr_complex>::make(4);
    }
    {
        this->blocks_vector_sink_1 = blocks::vector_sink<gr_complex>::make(8);
    }
    {
	this->blocks_fft_0 = dash::fft::make(true, 4); // Forward, size 4
    }
    {
	this->blocks_fft_1 = dash::fft::make(false, 8); // Reverse, size 8
    }

// Connections:
    this->tb->hier_block2::connect(this->blocks_vector_source_0, 0, this->blocks_fft_0, 0);
    this->tb->hier_block2::connect(this->blocks_fft_0, 0, this->blocks_vector_sink_0, 0);
    this->tb->hier_block2::connect(this->blocks_vector_source_1, 0, this->blocks_fft_1, 0);
    this->tb->hier_block2::connect(this->blocks_fft_1, 0, this->blocks_vector_sink_1, 0);
}

horn::~horn () {
}

// Callbacks:
int horn::get_samp_rate () const {
    return this->samp_rate;
}

void horn::set_samp_rate (int samp_rate) {
    this->samp_rate = samp_rate;
}


int main (int argc, char **argv) {

    horn* top_block = new horn();
    /*top_block->tb->start();
    std::cout << "Press Enter to quit: ";
    std::cin.ignore();
    top_block->tb->stop();
    top_block->tb->wait();*/
    top_block->tb->run();
    std::vector<gr_complex> test_data;
    test_data = top_block->blocks_vector_sink_0->data();
    std::vector<gr_complex> test_data2;
    test_data2 = top_block->blocks_vector_sink_1->data();
    // FFT'd Data
    printf("Forward FFT\n");
    for(gr_complex i : test_data){
        printf("%f, %f\n", i.real(), i.imag());
    }
    printf("Reverse FFT\n");
    for(gr_complex i : test_data2){
        printf("%f, %f\n", i.real(), i.imag());
    }
    return 0;
}
