#include<detection.c>
#include<decode.c>
#include<baseband_lib.c>
void wifirx_fields_cleanup(struct wifirx_fields *wifirx_param){


}

void wifirx_nop(struct task_nodes *task){
	//pthread_t self;
	//self = pthread_self();
	//printf("Starting wifi rx app id %d task id %d task name %s Core ID %lu\n",task->app_id, task->task_id, task->task_name, self);
	//printf("Ending wifi rx app id %d task id %d task name %s Core ID %lu\n",task->app_id, task->task_id, task->task_name, self);
}


void wifirx_init(struct struct_app *wifirx, struct wifirx_fields *wifirx_param ){

	strcpy(wifirx->app_name,"WiFi RX");
	wifirx->task_count = (9*SYM_NUM) + 1;
	struct task_nodes *nodes = malloc(wifirx->task_count * sizeof(struct task_nodes));
	wifirx->head_node = nodes;

    viterbiDecoder_t *vDecoder;
    vDecoder = wifirx_param->vDecoder;

    init_viterbiDecoder(vDecoder);
    wifirx_param->dId = get_viterbiDecoder(vDecoder);
    set_viterbiDecoder(wifirx_param->dId, vDecoder);

    // clean RX buffer
    for(int j=0; j<SYM_NUM*USR_DAT_LEN; j++) wifirx_param->descram[j] = 0;

    wifirx_param->maxIndex = 0; 
    wifirx_param->prtMaxIdx = 0;
    wifirx_param->payloadStart = 0;
    wifirx_param->payloadPrtStart = 0;
    wifirx_param->spRate = 1;
    wifirx_param->sIdx = 0;
    wifirx_param->sampleCount = 0;
    wifirx_param->maxIndex = 0;
    
    frameDetection(wifirx_param);

    wifirx_param->fft_n = FFT_N;
    wifirx_param->hw_fft_busy = 1;
    wifirx_param->pilot_len = FFT_N;
    wifirx_param->demod_n = INPUT_LEN;
    wifirx_param->deinterleaver_n = OUTPUT_LEN;
    wifirx_param->rate = PUNC_RATE_1_2;
    wifirx_param->descrambler_n = USR_DAT_LEN;
    wifirx_param->sym_num = 0;


	for (int i = 0;i<SYM_NUM;i++){
		nodes[9*i+0].task_id = 9*i+0;
		nodes[9*i+0].app_id = wifirx->app_id;
		nodes[9*i+0].succ_count = 1;
		nodes[9*i+0].succ = malloc(nodes[9*i+0].succ_count * sizeof(struct task_nodes*));
		if (i==0){
			nodes[9*i+0].pred_count = 0;
		}
		else{
			nodes[9*i+0].pred_count = 1;

		}
		nodes[9*i+0].pred = malloc(nodes[9*i+0].pred_count * sizeof(struct task_nodes*));
		{
			char tmp[50]= "PAYLOAD_EXTRACTION_";
			char tmp1[50];
			sprintf(tmp1,"%d",i);
			strcat(tmp,tmp1);
			
			strcpy(nodes[9*i+0].task_name,tmp);
		}
		nodes[9*i+0].fields =wifirx_param;
		nodes[9*i+0].run_func = payloadExt;
		nodes[9*i+0].complete_flag = 0;
		nodes[9*i+0].running_flag = 0;
		nodes[9*i+0].supported_resource_count = 1;
		nodes[9*i+0].supported_resources = malloc(nodes[9*i+0].supported_resource_count*sizeof(char *));
		nodes[9*i+0].estimated_execution = malloc(nodes[9*i+0].supported_resource_count*sizeof(float));
		for (int j=0; j< nodes[9*i+0].supported_resource_count;j++){
			nodes[9*i+0].supported_resources[j]= malloc(25*sizeof(char));
		}
		strcpy(nodes[9*i+0].supported_resources[0],"cpu");
		nodes[9*i+0].estimated_execution[0] = 100.0;


		nodes[9*i+0].alloc_resource_config_input = -1;
		if (i!= 0){
			nodes[9*i+0].pred[0] = &(nodes[9*(i-1)+8]);	
		}
		nodes[9*i+0].succ[0] = &(nodes[9*i+1]);	





		//"FFT"
		nodes[9*i+1].task_id = 9*i+1;
		nodes[9*i+1].app_id = wifirx->app_id;
		nodes[9*i+1].succ_count = 1;
		nodes[9*i+1].succ = malloc(nodes[9*i+1].succ_count * sizeof(struct task_nodes*));
		nodes[9*i+1].pred_count = 1;
		nodes[9*i+1].pred = malloc(nodes[9*i+1].pred_count * sizeof(struct task_nodes*));
		{
			char tmp[50]= "FFT_";
			char tmp1[50];
			sprintf(tmp1,"%d",i);
			strcat(tmp,tmp1);
			strcpy(nodes[9*i+1].task_name,tmp);
		}
		nodes[9*i+1].fields =wifirx_param;
		nodes[9*i+1].run_func = fft_hs;
		nodes[9*i+1].complete_flag = 0;
		nodes[9*i+1].running_flag = 0;
		nodes[9*i+1].supported_resource_count = 2;
		nodes[9*i+1].supported_resources = malloc(nodes[9*i+1].supported_resource_count*sizeof(char *));
		nodes[9*i+1].estimated_execution = malloc(nodes[9*i+1].supported_resource_count*sizeof(float));
		for (int j=0; j< nodes[9*i+1].supported_resource_count;j++){
			nodes[9*i+1].supported_resources[j]= malloc(25*sizeof(char));
		}
		strcpy(nodes[9*i+1].supported_resources[0],"fft");
		nodes[9*i+1].estimated_execution[0] = 30.0;
		strcpy(nodes[9*i+1].supported_resources[1],"cpu");
		nodes[9*i+1].estimated_execution[1] = 100.0;


		nodes[9*i+1].alloc_resource_config_input = -1;
		nodes[9*i+1].pred[0] = &(nodes[9*i+0]);	
		nodes[9*i+1].succ[0] = &(nodes[9*i+2]);	




		//"Pilot_Extraction"
		nodes[9*i+2].task_id = 9*i+2;
		nodes[9*i+2].app_id = wifirx->app_id;
		nodes[9*i+2].succ_count = 1;
		nodes[9*i+2].succ = malloc(nodes[9*i+2].succ_count * sizeof(struct task_nodes*));
		nodes[9*i+2].pred_count = 1;
		nodes[9*i+2].pred = malloc(nodes[9*i+2].pred_count * sizeof(struct task_nodes*));
		{
			char tmp[50]= "Pilot_Extraction_";
			char tmp1[50];
			sprintf(tmp1,"%d",i);
			strcat(tmp,tmp1);
			strcpy(nodes[9*i+2].task_name,tmp);
		}
		nodes[9*i+2].fields =wifirx_param;
		nodes[9*i+2].run_func = pilotExtract;
		nodes[9*i+2].complete_flag = 0;
		nodes[9*i+2].running_flag = 0;
		nodes[9*i+2].supported_resource_count = 1;
		nodes[9*i+2].supported_resources = malloc(nodes[9*i+2].supported_resource_count*sizeof(char *));
		nodes[9*i+2].estimated_execution = malloc(nodes[9*i+2].supported_resource_count*sizeof(float));
		for (int j=0; j< nodes[9*i+2].supported_resource_count;j++){
			nodes[9*i+2].supported_resources[j]= malloc(25*sizeof(char));
		}
		strcpy(nodes[9*i+2].supported_resources[0],"cpu");
		nodes[9*i+2].estimated_execution[0] = 100.0;


		nodes[9*i+2].alloc_resource_config_input = -1;
		nodes[9*i+2].pred[0] = &(nodes[9*i+1]);	
		nodes[9*i+2].succ[0] = &(nodes[9*i+3]);	






		//"Pilot_Removal"
		nodes[9*i+3].task_id = 9*i+3;
		nodes[9*i+3].app_id = wifirx->app_id;
		nodes[9*i+3].succ_count = 1;
		nodes[9*i+3].succ = malloc(nodes[9*i+3].succ_count * sizeof(struct task_nodes*));
		nodes[9*i+3].pred_count = 1;
		nodes[9*i+3].pred = malloc(nodes[9*i+3].pred_count * sizeof(struct task_nodes*));
		{
			char tmp[50]= "Pilot_Removal_";
			char tmp1[50];
			sprintf(tmp1,"%d",i);
			strcat(tmp,tmp1);
			strcpy(nodes[9*i+3].task_name,tmp);
		}
		nodes[9*i+3].fields =wifirx_param;
		nodes[9*i+3].run_func = pilotRemove;
		nodes[9*i+3].complete_flag = 0;
		nodes[9*i+3].running_flag = 0;
		nodes[9*i+3].supported_resource_count = 1;
		nodes[9*i+3].supported_resources = malloc(nodes[9*i+3].supported_resource_count*sizeof(char *));
		nodes[9*i+3].estimated_execution = malloc(nodes[9*i+3].supported_resource_count*sizeof(float));
		for (int j=0; j< nodes[9*i+3].supported_resource_count;j++){
			nodes[9*i+3].supported_resources[j]= malloc(25*sizeof(char));
		}
		strcpy(nodes[9*i+3].supported_resources[0],"cpu");
		nodes[9*i+3].estimated_execution[0] = 100.0;


		nodes[9*i+3].alloc_resource_config_input = -1;
		nodes[9*i+3].pred[0] = &(nodes[9*i+2]);	
		nodes[9*i+3].succ[0] = &(nodes[9*i+4]);	






		//"QPSK_Demod"
		nodes[9*i+4].task_id = 9*i+4;
		nodes[9*i+4].app_id = wifirx->app_id;
		nodes[9*i+4].succ_count = 1;
		nodes[9*i+4].succ = malloc(nodes[9*i+4].succ_count * sizeof(struct task_nodes*));
		nodes[9*i+4].pred_count = 1;
		nodes[9*i+4].pred = malloc(nodes[9*i+4].pred_count * sizeof(struct task_nodes*));
		{
			char tmp[50]= "QPSK_Demod_";
			char tmp1[50];
			sprintf(tmp1,"%d",i);
			strcat(tmp,tmp1);
			strcpy(nodes[9*i+4].task_name,tmp);
		}
		nodes[9*i+4].fields =wifirx_param;
		nodes[9*i+4].run_func = DeMOD_QPSK; 
		nodes[9*i+4].complete_flag = 0;
		nodes[9*i+4].running_flag = 0;
		nodes[9*i+4].supported_resource_count = 1;
		nodes[9*i+4].supported_resources = malloc(nodes[9*i+4].supported_resource_count*sizeof(char *));
		nodes[9*i+4].estimated_execution = malloc(nodes[9*i+4].supported_resource_count*sizeof(float));
		for (int j=0; j< nodes[9*i+4].supported_resource_count;j++){
			nodes[9*i+4].supported_resources[j]= malloc(25*sizeof(char));
		}
		strcpy(nodes[9*i+4].supported_resources[0],"cpu");
		nodes[9*i+4].estimated_execution[0] = 100.0;


		nodes[9*i+4].alloc_resource_config_input = -1;
		nodes[9*i+4].pred[0] = &(nodes[9*i+3]);	
		nodes[9*i+4].succ[0] = &(nodes[9*i+5]);	






		// "Deinteterleaver"
		nodes[9*i+5].task_id = 9*i+5;
		nodes[9*i+5].app_id = wifirx->app_id;
		nodes[9*i+5].succ_count = 1;
		nodes[9*i+5].succ = malloc(nodes[9*i+5].succ_count * sizeof(struct task_nodes*));
		nodes[9*i+5].pred_count = 1;
		nodes[9*i+5].pred = malloc(nodes[9*i+5].pred_count * sizeof(struct task_nodes*));
		{
			char tmp[50]= "Deinteterleaver_";
			char tmp1[50];
			sprintf(tmp1,"%d",i);
			strcat(tmp,tmp1);
			strcpy(nodes[9*i+5].task_name,tmp);
		}
		nodes[9*i+5].fields =wifirx_param;
		nodes[9*i+5].run_func = deinterleaver;
		nodes[9*i+5].complete_flag = 0;
		nodes[9*i+5].running_flag = 0;
		nodes[9*i+5].supported_resource_count = 1;
		nodes[9*i+5].supported_resources = malloc(nodes[9*i+5].supported_resource_count*sizeof(char *));
		nodes[9*i+5].estimated_execution = malloc(nodes[9*i+5].supported_resource_count*sizeof(float));
		for (int j=0; j< nodes[9*i+5].supported_resource_count;j++){
			nodes[9*i+5].supported_resources[j]= malloc(25*sizeof(char));
		}
		strcpy(nodes[9*i+5].supported_resources[0],"cpu");
		nodes[9*i+5].estimated_execution[0] = 100.0;


		nodes[9*i+5].alloc_resource_config_input = -1;
		nodes[9*i+5].pred[0] = &(nodes[9*i+4]);	
		nodes[9*i+5].succ[0] = &(nodes[9*i+6]);	




		// Format_Compression
		nodes[9*i+6].task_id = 9*i+6;
		nodes[9*i+6].app_id = wifirx->app_id;
		nodes[9*i+6].succ_count = 1;
		nodes[9*i+6].succ = malloc(nodes[9*i+6].succ_count * sizeof(struct task_nodes*));
		nodes[9*i+6].pred_count = 1;
		nodes[9*i+6].pred = malloc(nodes[9*i+6].pred_count * sizeof(struct task_nodes*));
		{
			char tmp[50]= "Format_Compression_";
			char tmp1[50];
			sprintf(tmp1,"%d",i);
			strcat(tmp,tmp1);
			strcpy(nodes[9*i+6].task_name,tmp);
		}
		nodes[9*i+6].fields =wifirx_param;
		nodes[9*i+6].run_func = formatConversion;
		nodes[9*i+6].complete_flag = 0;
		nodes[9*i+6].running_flag = 0;
		nodes[9*i+6].supported_resource_count = 1;
		nodes[9*i+6].supported_resources = malloc(nodes[9*i+6].supported_resource_count*sizeof(char *));
		nodes[9*i+6].estimated_execution = malloc(nodes[9*i+6].supported_resource_count*sizeof(float));
		for (int j=0; j< nodes[9*i+6].supported_resource_count;j++){
			nodes[9*i+6].supported_resources[j]= malloc(25*sizeof(char));
		}
		strcpy(nodes[9*i+6].supported_resources[0],"cpu");
		nodes[9*i+6].estimated_execution[0] = 100.0;


		nodes[9*i+6].alloc_resource_config_input = -1;
		nodes[9*i+6].pred[0] = &(nodes[9*i+5]);	
		nodes[9*i+6].succ[0] = &(nodes[9*i+7]);	






		// Decoder
		nodes[9*i+7].task_id = 9*i+7;
		nodes[9*i+7].app_id = wifirx->app_id;
		nodes[9*i+7].succ_count = 1;
		nodes[9*i+7].succ = malloc(nodes[9*i+7].succ_count * sizeof(struct task_nodes*));
		nodes[9*i+7].pred_count = 1;
		nodes[9*i+7].pred = malloc(nodes[9*i+7].pred_count * sizeof(struct task_nodes*));
		{
			char tmp[50]= "Decoder_";
			char tmp1[50];
			sprintf(tmp1,"%d",i);
			strcat(tmp,tmp1);
			strcpy(nodes[9*i+7].task_name,tmp);
		}
		nodes[9*i+7].fields =wifirx_param;
		nodes[9*i+7].run_func = viterbi_decoding;
		nodes[9*i+7].complete_flag = 0;
		nodes[9*i+7].running_flag = 0;
		nodes[9*i+7].supported_resource_count = 1;
		nodes[9*i+7].supported_resources = malloc(nodes[9*i+7].supported_resource_count*sizeof(char *));
		nodes[9*i+7].estimated_execution = malloc(nodes[9*i+7].supported_resource_count*sizeof(float));
		for (int j=0; j< nodes[9*i+7].supported_resource_count;j++){
			nodes[9*i+7].supported_resources[j]= malloc(25*sizeof(char));
		}
		strcpy(nodes[9*i+7].supported_resources[0],"cpu");
		nodes[9*i+7].estimated_execution[0] = 100.0;


		nodes[9*i+7].alloc_resource_config_input = -1;
		nodes[9*i+7].pred[0] = &(nodes[9*i+6]);	
		nodes[9*i+7].succ[0] = &(nodes[9*i+8]);	





		// Descrambler
		nodes[9*i+8].task_id = 9*i+8;
		nodes[9*i+8].app_id = wifirx->app_id;
		nodes[9*i+8].succ_count = 1;
		nodes[9*i+8].succ = malloc(nodes[9*i+8].succ_count * sizeof(struct task_nodes*));
		nodes[9*i+8].pred_count = 1;
		nodes[9*i+8].pred = malloc(nodes[9*i+8].pred_count * sizeof(struct task_nodes*));
		{
			char tmp[50]= "Descrambler_";
			char tmp1[50];
			sprintf(tmp1,"%d",i);
			strcat(tmp,tmp1);
			strcpy(nodes[9*i+8].task_name,tmp);
		}
		nodes[9*i+8].fields =wifirx_param;
		nodes[9*i+8].run_func = descrambler;
		nodes[9*i+8].complete_flag = 0;
		nodes[9*i+8].running_flag = 0;
		nodes[9*i+8].supported_resource_count = 1;
		nodes[9*i+8].supported_resources = malloc(nodes[9*i+8].supported_resource_count*sizeof(char *));
		nodes[9*i+8].estimated_execution = malloc(nodes[9*i+8].supported_resource_count*sizeof(float));
		for (int j=0; j< nodes[9*i+8].supported_resource_count;j++){
			nodes[9*i+8].supported_resources[j]= malloc(25*sizeof(char));
		}
		strcpy(nodes[9*i+8].supported_resources[0],"cpu");
		nodes[9*i+8].estimated_execution[0] = 100.0;


		nodes[9*i+8].alloc_resource_config_input = -1;
		nodes[9*i+8].pred[0] = &(nodes[9*i+7]);	
		nodes[9*i+8].succ[0] = &(nodes[9*i+9]);	


	}




	//"POST-PROC"
	nodes[9*SYM_NUM].task_id = 9*SYM_NUM;
	nodes[9*SYM_NUM].app_id = wifirx->app_id;
	nodes[9*SYM_NUM].succ_count = 0;
	nodes[9*SYM_NUM].succ = malloc(nodes[9*SYM_NUM].succ_count * sizeof(struct task_nodes*));
	nodes[9*SYM_NUM].pred_count = 1;
	nodes[9*SYM_NUM].pred = malloc(nodes[9*SYM_NUM].pred_count * sizeof(struct task_nodes*));
	strcpy(nodes[9*SYM_NUM].task_name,"POST-PROC");
	nodes[9*SYM_NUM].fields =wifirx_param;
	//nodes[9*SYM_NUM].run_func = messagedecoder;
	nodes[9*SYM_NUM].run_func= wifirx_nop;
	nodes[9*SYM_NUM].complete_flag = 0;
	nodes[9*SYM_NUM].running_flag = 0;
	nodes[9*SYM_NUM].supported_resource_count = 1;
	nodes[9*SYM_NUM].supported_resources = malloc(nodes[9*SYM_NUM].supported_resource_count*sizeof(char *));
	nodes[9*SYM_NUM].estimated_execution = malloc(nodes[9*SYM_NUM].supported_resource_count*sizeof(float));
	for (int j=0; j< nodes[9*SYM_NUM].supported_resource_count;j++){
		nodes[9*SYM_NUM].supported_resources[j]= malloc(25*sizeof(char));
	}
	strcpy(nodes[9*SYM_NUM].supported_resources[0],"cpu");
	nodes[9*SYM_NUM].estimated_execution[0] = 100.0;


	nodes[9*SYM_NUM].alloc_resource_config_input = -1;
	nodes[9*SYM_NUM].pred[0] = &(nodes[9*SYM_NUM -1]);	





}

