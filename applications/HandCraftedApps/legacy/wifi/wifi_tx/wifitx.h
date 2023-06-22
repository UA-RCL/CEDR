#include<scrambler_descrambler.c>
#include<txData.c>
#include<viterbi.c>
#include<interleaver_deintleaver.c>
#include<qpsk_Mod_Demod.c>
#include<pilot.c>
#include<fft_hs.c>
#include<CyclicPrefix.c>
#include<post_processing.c>
#include<Preamble_ST_LG.c>
#include<crc.c>

extern FILE *txdata_file;
FILE *txdata_file;

void wifitx_fields_cleanup(struct wifitx_fields *wifitx_param) { }

void wifitx_nop(struct task_nodes *task){
	//pthread_t self;
	//self = pthread_self();
	//printf("Starting lag detect app id %d task id %d task name %s Core ID %lu\n",task->app_id, task->task_id, task->task_name, self);
	//printf("Ending lag detect app id %d task id %d task name %s Core ID %lu\n",task->app_id, task->task_id, task->task_name, self);
}


void wifitx_init(struct struct_app *wifitx, struct wifitx_fields *wifi_param ){

	strcpy(wifitx->app_name,"WiFi TX");
	wifitx->task_count = (7*SYM_NUM) + 1;
	struct task_nodes *nodes = malloc(wifitx->task_count * sizeof(struct task_nodes));
	wifitx->head_node = nodes;

    // WIFI Initialization
    //init_viterbiEncoder();
    //encoderId = get_viterbiEncoder();
    //set_viterbiEncoder(encoderId);
    
    int txOption;
    FILE *cfp;
    char buf[1024];

    cfp = fopen("tx.cfg", "r");
    if(cfp == NULL) {
       printf("fail to open config file\n");
       exit(1);
    }

    fgets(buf, 1024, cfp);
    sscanf(buf, "%d", &txOption);
    printf("- %s\n", (txOption == 0) ? "Tx fixed string" : "Tx variable string");

    //FILE *txdata_file = fopen("txdata_1.txt", "w");
    txdata_file = fopen("txdata_1.txt", "w");
    txDataGen(txOption, wifi_param->inbit, SYM_NUM);
    wifi_param->user_data_len = USR_DAT_LEN; //txDataGen(txOption, wifi_param->inbit, SYM_NUM);
    printf("user_data_len: %d\n", wifi_param->user_data_len);
    printf("USR_DAT_LEN: %d\n", USR_DAT_LEN);
    wifi_param->hw_fft_busy   = 1;
    wifi_param->encoder_id    = 1;
    wifi_param->output_len    = OUTPUT_LEN;
    wifi_param->fft_n         = FFT_N;
    wifi_param->cyclic_length = CYC_LEN;
    wifi_param->sym_num       = 0;
    
    // End of WIFI Initialization

	for (int i = 0;i<SYM_NUM;i++){
		nodes[7*i+0].task_id = 7*i+0;
		nodes[7*i+0].app_id = wifitx->app_id;
		nodes[7*i+0].succ_count = 1;
		nodes[7*i+0].succ = malloc(nodes[7*i+0].succ_count * sizeof(struct task_nodes*));
		if (i==0){
			nodes[7*i+0].pred_count = 0;
		}
		else{
			nodes[7*i+0].pred_count = 1;

		}
		nodes[7*i+0].pred = malloc(nodes[7*i+0].pred_count * sizeof(struct task_nodes*));
		{
			char tmp[50]= "SCRAMBLER_";
			char tmp1[50];
			sprintf(tmp1,"%d",i);
			strcat(tmp,tmp1);
			
			strcpy(nodes[7*i+0].task_name,tmp);
		}
		nodes[7*i+0].fields =wifi_param;
		nodes[7*i+0].run_func = scrambler;
		nodes[7*i+0].complete_flag = 0;
		nodes[7*i+0].running_flag = 0;
		nodes[7*i+0].supported_resource_count = 1;
		nodes[7*i+0].supported_resources = malloc(nodes[7*i+0].supported_resource_count*sizeof(char *));
		nodes[7*i+0].estimated_execution = malloc(nodes[7*i+0].supported_resource_count*sizeof(float));
		for (int j=0; j< nodes[7*i+0].supported_resource_count;j++){
			nodes[7*i+0].supported_resources[j]= malloc(25*sizeof(char));
		}
		strcpy(nodes[7*i+0].supported_resources[0],"cpu");
		nodes[7*i+0].estimated_execution[0] = 100.0;


		nodes[7*i+0].alloc_resource_config_input = -1;
		if (i!= 0){
			nodes[7*i+0].pred[0] = &(nodes[7*(i-1)+6]);	
		}
		nodes[7*i+0].succ[0] = &(nodes[7*i+1]);	





		//"ENCODER"
		nodes[7*i+1].task_id = 7*i+1;
		nodes[7*i+1].app_id = wifitx->app_id;
		nodes[7*i+1].succ_count = 1;
		nodes[7*i+1].succ = malloc(nodes[7*i+1].succ_count * sizeof(struct task_nodes*));
		nodes[7*i+1].pred_count = 1;
		nodes[7*i+1].pred = malloc(nodes[7*i+1].pred_count * sizeof(struct task_nodes*));
		{
			char tmp[50]= "ENCODER";
			char tmp1[50];
			sprintf(tmp1,"%d",i);
			strcat(tmp,tmp1);
			strcpy(nodes[7*i+1].task_name,tmp);
		}
		nodes[7*i+1].fields =wifi_param;
		nodes[7*i+1].run_func = viterbi_encoding;
		nodes[7*i+1].complete_flag = 0;
		nodes[7*i+1].running_flag = 0;
		nodes[7*i+1].supported_resource_count = 1;
		nodes[7*i+1].supported_resources = malloc(nodes[7*i+1].supported_resource_count*sizeof(char *));
		nodes[7*i+1].estimated_execution = malloc(nodes[7*i+1].supported_resource_count*sizeof(float));
		for (int j=0; j< nodes[7*i+1].supported_resource_count;j++){
			nodes[7*i+1].supported_resources[j]= malloc(25*sizeof(char));
		}
		strcpy(nodes[7*i+1].supported_resources[0],"cpu");
		nodes[7*i+1].estimated_execution[0] = 100.0;


		nodes[7*i+1].alloc_resource_config_input = -1;
		nodes[7*i+1].pred[0] = &(nodes[7*i+0]);	
		nodes[7*i+1].succ[0] = &(nodes[7*i+2]);	




		//"INTERLEAVER"
		nodes[7*i+2].task_id = 7*i+2;
		nodes[7*i+2].app_id = wifitx->app_id;
		nodes[7*i+2].succ_count = 1;
		nodes[7*i+2].succ = malloc(nodes[7*i+2].succ_count * sizeof(struct task_nodes*));
		nodes[7*i+2].pred_count = 1;
		nodes[7*i+2].pred = malloc(nodes[7*i+2].pred_count * sizeof(struct task_nodes*));
		{
			char tmp[50]= "INTERLEAVER";
			char tmp1[50];
			sprintf(tmp1,"%d",i);
			strcat(tmp,tmp1);
			strcpy(nodes[7*i+2].task_name,tmp);
		}
		nodes[7*i+2].fields =wifi_param;
		nodes[7*i+2].run_func = interleaver;
		nodes[7*i+2].complete_flag = 0;
		nodes[7*i+2].running_flag = 0;
		nodes[7*i+2].supported_resource_count = 1;
		nodes[7*i+2].supported_resources = malloc(nodes[7*i+2].supported_resource_count*sizeof(char *));
		nodes[7*i+2].estimated_execution = malloc(nodes[7*i+2].supported_resource_count*sizeof(float));
		for (int j=0; j< nodes[7*i+2].supported_resource_count;j++){
			nodes[7*i+2].supported_resources[j]= malloc(25*sizeof(char));
		}
		strcpy(nodes[7*i+2].supported_resources[0],"cpu");
		nodes[7*i+2].estimated_execution[0] = 100.0;


		nodes[7*i+2].alloc_resource_config_input = -1;
		nodes[7*i+2].pred[0] = &(nodes[7*i+1]);	
		nodes[7*i+2].succ[0] = &(nodes[7*i+3]);	






		//"QPSK"
		nodes[7*i+3].task_id = 7*i+3;
		nodes[7*i+3].app_id = wifitx->app_id;
		nodes[7*i+3].succ_count = 1;
		nodes[7*i+3].succ = malloc(nodes[7*i+3].succ_count * sizeof(struct task_nodes*));
		nodes[7*i+3].pred_count = 1;
		nodes[7*i+3].pred = malloc(nodes[7*i+3].pred_count * sizeof(struct task_nodes*));
		{
			char tmp[50]= "QPSK";
			char tmp1[50];
			sprintf(tmp1,"%d",i);
			strcat(tmp,tmp1);
			strcpy(nodes[7*i+3].task_name,tmp);
		}
		nodes[7*i+3].fields =wifi_param;
		nodes[7*i+3].run_func = MOD_QPSK;
		nodes[7*i+3].complete_flag = 0;
		nodes[7*i+3].running_flag = 0;
		nodes[7*i+3].supported_resource_count = 1;
		nodes[7*i+3].supported_resources = malloc(nodes[7*i+3].supported_resource_count*sizeof(char *));
		nodes[7*i+3].estimated_execution = malloc(nodes[7*i+3].supported_resource_count*sizeof(float));
		for (int j=0; j< nodes[7*i+3].supported_resource_count;j++){
			nodes[7*i+3].supported_resources[j]= malloc(25*sizeof(char));
		}
		strcpy(nodes[7*i+3].supported_resources[0],"cpu");
		nodes[7*i+3].estimated_execution[0] = 100.0;


		nodes[7*i+3].alloc_resource_config_input = -1;
		nodes[7*i+3].pred[0] = &(nodes[7*i+2]);	
		nodes[7*i+3].succ[0] = &(nodes[7*i+4]);	






		//"PILOT"
		nodes[7*i+4].task_id = 7*i+4;
		nodes[7*i+4].app_id = wifitx->app_id;
		nodes[7*i+4].succ_count = 1;
		nodes[7*i+4].succ = malloc(nodes[7*i+4].succ_count * sizeof(struct task_nodes*));
		nodes[7*i+4].pred_count = 1;
		nodes[7*i+4].pred = malloc(nodes[7*i+4].pred_count * sizeof(struct task_nodes*));
		{
			char tmp[50]= "PILOT";
			char tmp1[50];
			sprintf(tmp1,"%d",i);
			strcat(tmp,tmp1);
			strcpy(nodes[7*i+4].task_name,tmp);
		}
		nodes[7*i+4].fields =wifi_param;
		nodes[7*i+4].run_func = pilotInsertion;
		nodes[7*i+4].complete_flag = 0;
		nodes[7*i+4].running_flag = 0;
		nodes[7*i+4].supported_resource_count = 1;
		nodes[7*i+4].supported_resources = malloc(nodes[7*i+4].supported_resource_count*sizeof(char *));
		nodes[7*i+4].estimated_execution = malloc(nodes[7*i+4].supported_resource_count*sizeof(float));
		for (int j=0; j< nodes[7*i+4].supported_resource_count;j++){
			nodes[7*i+4].supported_resources[j]= malloc(25*sizeof(char));
		}
		strcpy(nodes[7*i+4].supported_resources[0],"cpu");
		nodes[7*i+4].estimated_execution[0] = 100.0;


		nodes[7*i+4].alloc_resource_config_input = -1;
		nodes[7*i+4].pred[0] = &(nodes[7*i+3]);	
		nodes[7*i+4].succ[0] = &(nodes[7*i+5]);	






		//"IFFT"
		nodes[7*i+5].task_id = 7*i+5;
		nodes[7*i+5].app_id = wifitx->app_id;
		nodes[7*i+5].succ_count = 1;
		nodes[7*i+5].succ = malloc(nodes[7*i+5].succ_count * sizeof(struct task_nodes*));
		nodes[7*i+5].pred_count = 1;
		nodes[7*i+5].pred = malloc(nodes[7*i+5].pred_count * sizeof(struct task_nodes*));
		{
			char tmp[50]= "IFFT";
			char tmp1[50];
			sprintf(tmp1,"%d",i);
			strcat(tmp,tmp1);
			strcpy(nodes[7*i+5].task_name,tmp);
		}
		nodes[7*i+5].fields =wifi_param;
		nodes[7*i+5].run_func = ifft_hs;
		nodes[7*i+5].complete_flag = 0;
		nodes[7*i+5].running_flag = 0;
		nodes[7*i+5].supported_resource_count = 2;
		nodes[7*i+5].supported_resources = malloc(nodes[7*i+5].supported_resource_count*sizeof(char *));
		nodes[7*i+5].estimated_execution = malloc(nodes[7*i+5].supported_resource_count*sizeof(float));
		for (int j=0; j< nodes[7*i+5].supported_resource_count;j++){
			nodes[7*i+5].supported_resources[j]= malloc(25*sizeof(char));
		}
		strcpy(nodes[7*i+5].supported_resources[0],"fft");
		nodes[7*i+5].estimated_execution[0] = 30.0;
		strcpy(nodes[7*i+5].supported_resources[1],"cpu");
		nodes[7*i+5].estimated_execution[1] = 100.0;


		nodes[7*i+5].alloc_resource_config_input = -1;
		nodes[7*i+5].pred[0] = &(nodes[7*i+4]);	
		nodes[7*i+5].succ[0] = &(nodes[7*i+6]);	




		//"CRC"
		nodes[7*i+6].task_id = 7*i+6;
		nodes[7*i+6].app_id = wifitx->app_id;
		nodes[7*i+6].succ_count = 1;
		nodes[7*i+6].succ = malloc(nodes[7*i+6].succ_count * sizeof(struct task_nodes*));
		nodes[7*i+6].pred_count = 1;
		nodes[7*i+6].pred = malloc(nodes[7*i+6].pred_count * sizeof(struct task_nodes*));
		{
			char tmp[50]= "CRC";
			char tmp1[50];
			sprintf(tmp1,"%d",i);
			strcat(tmp,tmp1);
			strcpy(nodes[7*i+6].task_name,tmp);
		}
		nodes[7*i+6].fields =wifi_param;
		nodes[7*i+6].run_func = cyclicPrefix;
		nodes[7*i+6].complete_flag = 0;
		nodes[7*i+6].running_flag = 0;
		nodes[7*i+6].supported_resource_count = 1;
		nodes[7*i+6].supported_resources = malloc(nodes[7*i+6].supported_resource_count*sizeof(char *));
		nodes[7*i+6].estimated_execution = malloc(nodes[7*i+6].supported_resource_count*sizeof(float));
		for (int j=0; j< nodes[7*i+6].supported_resource_count;j++){
			nodes[7*i+6].supported_resources[j]= malloc(25*sizeof(char));
		}
		strcpy(nodes[7*i+6].supported_resources[0],"cpu");
		nodes[7*i+6].estimated_execution[0] = 100.0;


		nodes[7*i+6].alloc_resource_config_input = -1;
		nodes[7*i+6].pred[0] = &(nodes[7*i+5]);	
		nodes[7*i+6].succ[0] = &(nodes[7*i+7]);	








	}




	//"POST-PROC"
	nodes[7*SYM_NUM].task_id = 7*SYM_NUM;
	nodes[7*SYM_NUM].app_id = wifitx->app_id;
	nodes[7*SYM_NUM].succ_count = 0;
	nodes[7*SYM_NUM].succ = malloc(nodes[7*SYM_NUM].succ_count * sizeof(struct task_nodes*));
	nodes[7*SYM_NUM].pred_count = 1;
	nodes[7*SYM_NUM].pred = malloc(nodes[7*SYM_NUM].pred_count * sizeof(struct task_nodes*));
	strcpy(nodes[7*SYM_NUM].task_name,"POST-PROC");
	nodes[7*SYM_NUM].fields =wifi_param;
	//nodes[7*SYM_NUM].run_func = post_processing;
	nodes[7*SYM_NUM].run_func = wifitx_nop;
	nodes[7*SYM_NUM].complete_flag = 0;
	nodes[7*SYM_NUM].running_flag = 0;
	nodes[7*SYM_NUM].supported_resource_count = 1;
	nodes[7*SYM_NUM].supported_resources = malloc(nodes[7*SYM_NUM].supported_resource_count*sizeof(char *));
	nodes[7*SYM_NUM].estimated_execution = malloc(nodes[7*SYM_NUM].supported_resource_count*sizeof(float));
	for (int j=0; j< nodes[7*SYM_NUM].supported_resource_count;j++){
		nodes[7*SYM_NUM].supported_resources[j]= malloc(25*sizeof(char));
	}
	strcpy(nodes[7*SYM_NUM].supported_resources[0],"cpu");
	nodes[7*SYM_NUM].estimated_execution[0] = 100.0;


	nodes[7*SYM_NUM].alloc_resource_config_input = -1;
	nodes[7*SYM_NUM].pred[0] = &(nodes[7*SYM_NUM -1]);	





}


