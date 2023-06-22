#include <cstring>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <dlfcn.h>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void generateJSON(void) {
	json output, DAG;
    unsigned int idx_offset = 0;
	int Nslow = 256;
    int Nfast = 512;

    //--- Application information
    output["AppName"] = "SAR_RDA";
    output["SharedObject"] = "SAR_RDA.so";
    output["Variables"] = json::object();
    	
/*********************************************************************************/

    //--- Head node
    DAG["S0"] = json::object();
    DAG["S0"]["platforms"] = json::array({
        {
            {"name", "cpu"},
            {"nodecost", 1.0f},
            {"runfunc", "SAR_node_head"}
        }
    });
    DAG["S0"]["predecessors"] = json::array();
    DAG["S0"]["successors"] = json::array();
    DAG["S0"]["task_id"] = idx_offset + 0;
    	
/*********************************************************************************/

    //--- LFM_1
    idx_offset += 1;
    DAG["LFM_1"] = json::object();
    DAG["LFM_1"]["platforms"] = json::array({
        {
            {"name", "cpu"},
            {"nodecost", 30.0f},
            {"runfunc", "SAR_RDA_LFM_1"}
        }
    });
	DAG["S0"]["successors"].push_back(json::object({
		{"name", "LFM_1"},
		{"edgecost", 0}
	}));
	DAG["LFM_1"]["predecessors"] = json::array({
		{
			{"name", "S0"},
			{"edgecost", 0}
		}
	});
    DAG["LFM_1"]["successors"] = json::array();
    DAG["LFM_1"]["task_id"] = idx_offset + 0;

/*********************************************************************************/

    //--- LFM_2
    idx_offset += 1;
    DAG["LFM_2"] = json::object();
    DAG["LFM_2"]["platforms"] = json::array({
        {
            {"name", "cpu"},
            {"nodecost", 30.0f},
            {"runfunc", "SAR_RDA_LFM_2"}
        }
    });
	DAG["LFM_1"]["successors"].push_back(json::object({
		{"name", "LFM_2"},
		{"edgecost", 0}
	}));
	DAG["LFM_2"]["predecessors"] = json::array({
		{
			{"name", "LFM_1"},
			{"edgecost", 0}
		}
	});
    DAG["LFM_2"]["successors"] = json::array();
    DAG["LFM_2"]["task_id"] = idx_offset + 0;
	
/*********************************************************************************/

    //--- FFT 1 for phase 1
    idx_offset += 1; // 3 -> S0, LFM_1, LFM_2
    for (int i = 0; i < Nslow; i++) {
        std::string node_key = "FFT_1_" + std::to_string(i);
        DAG[node_key] = json::object();
        DAG[node_key]["platforms"] = json::array({
            {
                {"name", "cpu"},
                {"nodecost", 30.0f},
                {"runfunc", "SAR_RDA_1_FFT_cpu"}
            }
        });

        DAG["LFM_2"]["successors"].push_back(json::object({
            {"name", node_key},
            {"edgecost", 0}
        }));		
        DAG[node_key]["predecessors"] = json::array({
            {
                {"name", "LFM_2"},
                {"edgecost", 0}
            }			
        });

        DAG[node_key]["successors"] = json::array();
        DAG[node_key]["task_id"] = idx_offset + i;
    }
    	
/*********************************************************************************/
	
	//--- FFT-Shift for phase 1
    idx_offset += Nslow; // 1*Nslow + 3, FFT_1_ made Nslow tasks
    for (int i = 0; i < Nslow; i++) {
        std::string node_key = "FFTSHIFT_1_" + std::to_string(i);
        DAG[node_key] = json::object();
        DAG[node_key]["platforms"] = json::array({
            {
                {"name", "cpu"},
                {"nodecost", 30.0f},
                {"runfunc", "SAR_RDA_1_FFTSHIFT"}
            }
        });

		// this is top level task id: FFT_1_(i)
		// which calls underlying SAR_RDA_gsl_fft_cpu or SAR_RDA_gsl_fft_acc stubs
		// where each has dynamic vars, inits of vars and calling the real kernel
        DAG["FFT_1_" + std::to_string(i)]["successors"].push_back(json::object({
            {"name", node_key},
            {"edgecost", 0}
        }));
        DAG[node_key]["predecessors"] = json::array({
            {
                {"name", "FFT_1_" + std::to_string(i)},
                {"edgecost", 0}
            }
        });

        DAG[node_key]["successors"] = json::array();
        DAG[node_key]["task_id"] = idx_offset + i;
    }
	
/*********************************************************************************/
	
    //--- Mul for phase 1
    idx_offset += Nslow; // 2*Nslow + 3
    for (int i = 0; i < Nslow; i++) {
        std::string node_key = "Mul_1_" + std::to_string(i);
        DAG[node_key] = json::object();
        DAG[node_key]["platforms"] = json::array({
            {
                {"name", "cpu"},
                {"nodecost", 30.0f},
                {"runfunc", "SAR_RDA_1_Mul"}
            }
        });

		// this is top level task id: FFTSHIFT_1_(i)
		// which calls underlying SAR_RDA_gsl_fft_cpu or SAR_RDA_gsl_fft_acc stubs
		// where each has dynamic vars, inits of vars and calling the real kernel
        DAG["FFTSHIFT_1_" + std::to_string(i)]["successors"].push_back(json::object({
            {"name", node_key},
            {"edgecost", 0}
        }));
        DAG[node_key]["predecessors"] = json::array({
            {
                {"name", "FFTSHIFT_1_" + std::to_string(i)},
                {"edgecost", 0}
            }
        });

        DAG[node_key]["successors"] = json::array();
        DAG[node_key]["task_id"] = idx_offset + i;
    }	
    	
/*********************************************************************************/
    //--- IFFT for phase 1
    idx_offset += Nslow; // 3*Nslow + 3
    for (int t = 0; t < Nslow; t++) {
        std::string node_key = "IFFT_1_" + std::to_string(t);
        DAG[node_key] = json::object();
        DAG[node_key]["platforms"] = json::array({
            {
                {"name", "cpu"},
                {"nodecost", 30.0f},
                {"runfunc", "SAR_RDA_1_IFFT_cpu"}
            }
        });

		// this is top level task id: FFTSHIFT_1_(i)
		// which calls underlying SAR_RDA_gsl_fft_cpu or SAR_RDA_gsl_fft_acc stubs
		// where each has dynamic vars, inits of vars and calling the real kernel
        DAG["Mul_1_" + std::to_string(t)]["successors"].push_back(json::object({
            {"name", node_key},
            {"edgecost", 0}
        }));
        DAG[node_key]["predecessors"] = json::array({
            {
                {"name", "Mul_1_" + std::to_string(t)},
                {"edgecost", 0}
            }
        });

        DAG[node_key]["successors"] = json::array();
        DAG[node_key]["task_id"] = idx_offset + t;
    }

    //--- SAR_RDA_Allign_1
    idx_offset += Nslow; // 6*Nfast + 4*Nslow + 3
    DAG["Allign_1"] = json::object();
    DAG["Allign_1"]["platforms"] = json::array({
        {
            {"name", "cpu"},
            {"nodecost", 30.0f},
            {"runfunc", "SAR_RDA_Allign_1"}
        }
    });

    DAG["Allign_1"]["predecessors"] = json::array();
    for (int i = 0; i < Nslow; i++) {
        DAG["IFFT_1_" + std::to_string(i)]["successors"].push_back(json::object({
            {"name", "Allign_1"},
            {"edgecost", 0}
        }));
        DAG["Allign_1"]["predecessors"].push_back(json::object({
            {"name", "IFFT_1_" + std::to_string(i)},
            {"edgecost", 0}
        }));
    }

    DAG["Allign_1"]["successors"] = json::array();
    DAG["Allign_1"]["task_id"] = idx_offset + 0;	
 

/*********************************************************************************/	
	
    //--- FFT for phase 2
    idx_offset += 1; // 4*Nslow + 3
    for (int i = 0; i < Nfast; i++) {
        std::string node_key = "FFT_2_" + std::to_string(i);
        DAG[node_key] = json::object();
        DAG[node_key]["platforms"] = json::array({
            {
                {"name", "cpu"},
                {"nodecost", 30.0f},
                {"runfunc", "SAR_RDA_2_FFT_cpu"}
            }
        });
		
        DAG["Allign_1"]["successors"].push_back(json::object({
            {"name", node_key},
            {"edgecost", 0}
        }));
        DAG[node_key]["predecessors"] = json::array({
            {
                {"name", "Allign_1"},
                {"edgecost", 0}
            }
        });

        DAG[node_key]["successors"] = json::array();
        DAG[node_key]["task_id"] = idx_offset + i;
    }

/*********************************************************************************/
	
	//--- FFT-Shift for phase 2
    idx_offset += Nfast; // 1*Nfast + 4*Nslow + 3
    for (int i = 0; i < Nfast; i++) {
        std::string node_key = "FFTSHIFT_2_" + std::to_string(i);
        DAG[node_key] = json::object();
        DAG[node_key]["platforms"] = json::array({
            {
                {"name", "cpu"},
                {"nodecost", 30.0f},
                {"runfunc", "SAR_RDA_2_FFTSHIFT"}
            }
        });

		// this is top level task id: FFT_1_(i)
		// which calls underlying SAR_RDA_gsl_fft_cpu or SAR_RDA_gsl_fft_acc stubs
		// where each has dynamic vars, inits of vars and calling the real kernel
        DAG["FFT_2_" + std::to_string(i)]["successors"].push_back(json::object({
            {"name", node_key},
            {"edgecost", 0}
        }));
        DAG[node_key]["predecessors"] = json::array({
            {
                {"name", "FFT_2_" + std::to_string(i)},
                {"edgecost", 0}
            }
        });

        DAG[node_key]["successors"] = json::array();
        DAG[node_key]["task_id"] = idx_offset + i;
    }		

/*********************************************************************************/

    //--- SAR_RDA_Allign_2
    idx_offset += Nfast; // 6*Nfast + 4*Nslow + 3
    DAG["Allign_2"] = json::object();
    DAG["Allign_2"]["platforms"] = json::array({
        {
            {"name", "cpu"},
            {"nodecost", 30.0f},
            {"runfunc", "SAR_RDA_Allign_2"}
        }
    });

    DAG["Allign_2"]["predecessors"] = json::array();
    for (int i = 0; i < Nfast; i++) {
        DAG["FFTSHIFT_2_" + std::to_string(i)]["successors"].push_back(json::object({
            {"name", "Allign_2"},
            {"edgecost", 0}
        }));
        DAG["Allign_2"]["predecessors"].push_back(json::object({
            {"name", "FFTSHIFT_2_" + std::to_string(i)},
            {"edgecost", 0}
        }));
    }

    DAG["Allign_2"]["successors"] = json::array();
    DAG["Allign_2"]["task_id"] = idx_offset + 0;	


	
/*********************************************************************************/
	
    //--- Mul for phase 3
    idx_offset += 1; // 2*Nfast + 4*Nslow + 3
    for (int i = 0; i < Nfast; i++) {
        std::string node_key = "Mul_3_" + std::to_string(i);
        DAG[node_key] = json::object();
        DAG[node_key]["platforms"] = json::array({
            {
                {"name", "cpu"},
                {"nodecost", 30.0f},
                {"runfunc", "SAR_RDA_3_Mul"}
            }
        });

		// this is top level task id: FFTSHIFT_1_(i)
		// which calls underlying SAR_RDA_gsl_fft_cpu or SAR_RDA_gsl_fft_acc stubs
		// where each has dynamic vars, inits of vars and calling the real kernel
        DAG["Allign_2"]["successors"].push_back(json::object({
            {"name", node_key},
            {"edgecost", 0}
        }));
        DAG[node_key]["predecessors"] = json::array({
            {
                {"name", "Allign_2"},
                {"edgecost", 0}
            }
        });

        DAG[node_key]["successors"] = json::array();
        DAG[node_key]["task_id"] = idx_offset + i;
    }

/*********************************************************************************/
	
    //--- IFFT for phase 3
    idx_offset += Nfast; // 3*Nfast + 4*Nslow + 3
    for (int i = 0; i < Nfast; i++) {
        std::string node_key = "IFFT_3_" + std::to_string(i);
        DAG[node_key] = json::object();
        DAG[node_key]["platforms"] = json::array({
            {
                {"name", "cpu"},
                {"nodecost", 30.0f},
                {"runfunc", "SAR_RDA_3_IFFT_cpu"}
            }
        });

        DAG["Mul_3_" + std::to_string(i)]["successors"].push_back(json::object({
            {"name", node_key},
            {"edgecost", 0}
        }));
        DAG[node_key]["predecessors"] = json::array({
            {
                {"name", "Mul_3_" + std::to_string(i)},
                {"edgecost", 0}
            }
        });

        DAG[node_key]["successors"] = json::array();
        DAG[node_key]["task_id"] = idx_offset + i;
    }	
    	
/*********************************************************************************/
	
	//--- FFT-Shift for phase 3
    idx_offset += Nfast; // 4*Nfast + 4*Nslow + 3
    for (int i = 0; i < Nfast; i++) {
        std::string node_key = "FFTSHIFT_3_" + std::to_string(i);
        DAG[node_key] = json::object();
        DAG[node_key]["platforms"] = json::array({
            {
                {"name", "cpu"},
                {"nodecost", 30.0f},
                {"runfunc", "SAR_RDA_3_FFTSHIFT"}
            }
        });

		// this is top level task id: FFT_1_(i)
		// which calls underlying SAR_RDA_gsl_fft_cpu or SAR_RDA_gsl_fft_acc stubs
		// where each has dynamic vars, inits of vars and calling the real kernel
        DAG["IFFT_3_" + std::to_string(i)]["successors"].push_back(json::object({
            {"name", node_key},
            {"edgecost", 0}
        }));
        DAG[node_key]["predecessors"] = json::array({
            {
                {"name", "IFFT_3_" + std::to_string(i)},
                {"edgecost", 0}
            }
        });

        DAG[node_key]["successors"] = json::array();
        DAG[node_key]["task_id"] = idx_offset + i;
    }
	
/*********************************************************************************/
	
	//---Amplitude for phase 3
    idx_offset += Nfast; // 5*Nfast + 4*Nslow + 3
    for (int i = 0; i < Nfast; i++) {
        std::string node_key = "Amplitude_" + std::to_string(i);
        DAG[node_key] = json::object();
        DAG[node_key]["platforms"] = json::array({
            {
                {"name", "cpu"},
                {"nodecost", 30.0f},
                {"runfunc", "SAR_RDA_3_Amplitude"}
            }
        });

		// this is top level task id: FFT_1_(i)
		// which calls underlying SAR_RDA_gsl_fft_cpu or SAR_RDA_gsl_fft_acc stubs
		// where each has dynamic vars, inits of vars and calling the real kernel
        DAG["FFTSHIFT_3_" + std::to_string(i)]["successors"].push_back(json::object({
            {"name", node_key},
            {"edgecost", 0}
        }));
        DAG[node_key]["predecessors"] = json::array({
            {
                {"name", "FFTSHIFT_3_" + std::to_string(i)},
                {"edgecost", 0}
            }
        });

        DAG[node_key]["successors"] = json::array();
        DAG[node_key]["task_id"] = idx_offset + i;
    }	

/*********************************************************************************/

    //--- File Write
    idx_offset += Nfast; // 6*Nfast + 4*Nslow + 3
    DAG["FWrite"] = json::object();
    DAG["FWrite"]["platforms"] = json::array({
        {
            {"name", "cpu"},
            {"nodecost", 30.0f},
            {"runfunc", "SAR_RDA_FWrite"}
        }
    });

    DAG["FWrite"]["predecessors"] = json::array();
    for (int i = 0; i < Nfast; i++) {
        DAG["Amplitude_" + std::to_string(i)]["successors"].push_back(json::object({
            {"name", "FWrite"},
            {"edgecost", 0}
        }));
        DAG["FWrite"]["predecessors"].push_back(json::object({
            {"name", "Amplitude_" + std::to_string(i)},
            {"edgecost", 0}
        }));
    }

    DAG["FWrite"]["successors"] = json::array();
    DAG["FWrite"]["task_id"] = idx_offset + 0;	
	

    output["DAG"] = DAG;
    std::ofstream output_file("SAR_RDA.json");
    if (!output_file.is_open()) {
        fprintf(stderr, "Failed to open output file for writing JSON\n");
        exit(1);
    }
    output_file << std::setw(2) << output;
}

int main(void) {
    generateJSON();
}