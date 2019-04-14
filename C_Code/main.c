#include<stdio.h>
#include "OSLW_include.h"

//#include "style_trans_we_star.h"
//#include "style_trans_we_wave.h"
#include "style_trans_we_sumiao.h"
//#include "style_trans_we_shuimo.h"
//#include "style_trans_we_bitflow.h"



LwMatColType in_shape_list[4] = { 1,3,512,512 };
LwMatColType in_pad_list[4] = { 10,10,10,10 };

ParaType mean_list1[64];
ParaType var_list1[64];


ParaType mean_list2[128];
ParaType var_list2[128];


ParaType pool_out[128][133][133];

ParaType conv_out[128][266][266];


void Mat2Csv(ParaType *p, lw_u32 row, lw_u32 col, char *name)
{
	lw_u32  i, j;
	FILE *file;

	file = fopen(name, "w+");
	fprintf(file, "%d\n", row);
	fprintf(file, "%d\n", col);
	for ( i = 0; i < row; i++)
	{

		for ( j = 0; j < col; j++)
		{

			fprintf(file, "%f\n", *p);
			p++;
		}

	}

	fclose(file);
}


OSlwToolBPnnSTU st_nn;
OSLW_MEM_SIMPLE_DEF(mem, 32, 3000)


ParaType buf0[24][1024][1024], buf1[3][1024][1024], buf2[24][1024][1024];
ParaType *pp0 = (void *)buf0, *pp1 = (void *)buf1, *pp2 = (void *)buf2;
ParaType fast_cnn_temp[1024 * 1024 * 24];

void *StyleTransConvAppend(
	OSlwToolBPnnSTU *pnn,
	lw_u16 inx, lw_u16 iny, lw_u16 inz,
	lw_u16 kernx, lw_u16 kerny, lw_u16 kernc,
	lw_u16 move_l,
	ParaType *pin, ParaType *pout,
	ParaType *we, ParaType *gamma, ParaType *beta,
	OSlwToolNNLayerActFunSTU *pACT,
	OSlwMemoryBasicSTU *pmem,
	lw_u32 info[4]
)
{
	OSlwToolDListNodeSTU *pdl;
	OSlwToolNNLayerFullConSTU **pfcl;

	LwMatColType *shape_list;
	LwMatColType *pad_list;

	pdl = pmem->Calloc(pmem, sizeof(OSlwToolDListNodeSTU));

	shape_list = pmem->Calloc(pmem, 4 * sizeof(LwMatColType));
	pad_list = pmem->Calloc(pmem, 4 * sizeof(LwMatColType));

	shape_list[0] = 1;
	shape_list[1] = inz;
	shape_list[2] = iny;
	shape_list[3] = inx;

	pad_list[1] = pad_list[0] = kernx / 2;
	pad_list[2] = pad_list[3] = kernx / 2;


	pfcl = pmem->Calloc(pmem, sizeof(OSlwToolNNLayerFullConSTU *) * 4);

	pfcl[0] = OSlwToolNNLayerPadNew(
		pin, pout,
		4,
		shape_list, pad_list,
		1,
		OSlwToolNNPad_Constant,
		pmem,
		info
	);

	pfcl[1] = OSlwToolNNLayerConvNew(
		pout, pin,
		info[0], info[1], info[2],
		kernx, kerny, kernc,
		move_l, OSlwToolMatrixConvMethod_Valid,
		1,
		pmem,
		info
	);

	OSlwToolNNLayerConvSetIm2Col(
		(void *)(pfcl[1]),
		sizeof(fast_cnn_temp),
		fast_cnn_temp
	);

	pfcl[1]->Weight.a = we;


	pfcl[2] = OSlwToolNNLayerINormNew(
		pin, pout,
		NULL, NULL,
		info[0], info[1], info[2],
		1,
		pmem
	);

	pfcl[2]->Weight.a = gamma;
	pfcl[2]->Bias.a = beta;

	pfcl[3] = OSlwToolNNLayerActFunNew(
		pout, pout,
		info[3],
		1,
		pmem,
		pACT, 0
	);


	OSlwToolBPnnLayerAppend(pnn, pdl, 4, pfcl);
	return pfcl;
}

void * StyleTransResAppend(
	OSlwToolBPnnSTU *pnn,
	lw_u16 inx, lw_u16 iny, lw_u16 inz,
	lw_u16 kernx, lw_u16 kerny, lw_u16 kernc,
	ParaType *pin, ParaType *ptemp, ParaType *pout,
	ParaType *we1, ParaType *we2,
	OSlwMemoryBasicSTU *pmem,
	lw_u32 info[4]
)
{
	OSlwToolDListNodeSTU *pdl;
	OSlwToolNNLayerFullConSTU **pfcl;

	LwMatColType *shape_list1, *shape_list2, *shape_list3;
	LwMatColType *pad_list1, *pad_list2, *mix_list;

	ParaType **out_list;

	pdl = pmem->Calloc(pmem, sizeof(OSlwToolDListNodeSTU));
	pfcl = pmem->Calloc(pmem, sizeof(OSlwToolNNLayerFullConSTU *) * 7);

	shape_list1 = pmem->Calloc(pmem, 4 * sizeof(LwMatColType));
	shape_list2 = pmem->Calloc(pmem, 4 * sizeof(LwMatColType));
	shape_list3 = pmem->Calloc(pmem, 4 * sizeof(LwMatColType));

	pad_list1 = pmem->Calloc(pmem, 4 * sizeof(LwMatColType));
	pad_list2 = pmem->Calloc(pmem, 4 * sizeof(LwMatColType));
	mix_list = pmem->Calloc(pmem, 4 * 2 * 2 * sizeof(LwMatColType));


	out_list = pmem->Calloc(pmem, 2 * sizeof(ParaType *));

	shape_list3[0] = shape_list2[0] = shape_list1[0] = 1;
	shape_list3[1] = shape_list2[1] = shape_list1[1] = inz;
	shape_list3[2] = shape_list2[2] = shape_list1[2] = iny;
	shape_list3[3] = shape_list2[3] = shape_list1[3] = inx;

	pad_list1[1] = pad_list1[0] = pad_list2[1] = pad_list2[0] = kernx / 2;
	pad_list1[2] = pad_list1[3] = pad_list2[2] = pad_list2[3] = kerny / 2;

	pfcl[0] = OSlwToolNNLayerPadNew(
		pin, ptemp,
		4,
		shape_list1, pad_list1,
		1,
		OSlwToolNNPad_Constant,
		pmem,
		info
	);

	pfcl[1] = OSlwToolNNLayerConvNew(
		ptemp, pout,
		info[0], info[1], info[2],
		kernx, kerny, kernc,
		1, OSlwToolMatrixConvMethod_Valid,
		1,
		pmem,
		info
	);

	OSlwToolNNLayerConvSetIm2Col(
		(void *)(pfcl[1]),
		sizeof(fast_cnn_temp),
		fast_cnn_temp
	);

	pfcl[1]->Weight.a = we1;

	pfcl[2] = OSlwToolNNLayerActFunNew(
		pout, ptemp,
		info[3],
		1,
		pmem,
		LwReLU, 0
	);


	pfcl[3] = OSlwToolNNLayerPadNew(
		ptemp, pout,
		4,
		shape_list2, pad_list2,
		1,
		OSlwToolNNPad_Constant,
		pmem,
		info
	);

	pfcl[4] = OSlwToolNNLayerConvNew(
		pout, ptemp,
		info[0], info[1], info[2],
		kernx, kerny, kernc,
		1, OSlwToolMatrixConvMethod_Valid,
		1,
		pmem,
		info
	);

	OSlwToolNNLayerConvSetIm2Col(
		(void *)(pfcl[4]),
		sizeof(fast_cnn_temp),
		fast_cnn_temp
	);

	pfcl[4]->Weight.a = we2;


	out_list[0] = pin;
	out_list[1] = ptemp;

	mix_list[1] = 1;
	mix_list[3] = inz;
	mix_list[5] = iny;
	mix_list[7] = inx;

	mix_list[1 + 8] = 1;
	mix_list[3 + 8] = inz;
	mix_list[5 + 8] = iny;
	mix_list[7 + 8] = inx;

	pfcl[5] = OSlwToolNNLayerMixNew(
		out_list,
		pout,
		4,
		shape_list3,
		2,
		mix_list,
		1,
		pmem
	);

	OSlwToolBPnnLayerAppend(pnn, pdl, 6, pfcl);

	return pfcl;
}

void * StyleTransExtConvAppend(
	OSlwToolBPnnSTU *pnn,
	lw_u16 inx, lw_u16 iny, lw_u16 inz,
	lw_u16 kernx, lw_u16 kerny, lw_u16 kernc,
	lw_u16 move_l,
	ParaType *pin, ParaType *pout,
	ParaType *we, ParaType *gamma, ParaType *beta,
	OSlwToolNNLayerActFunSTU *pACT,
	OSlwMemoryBasicSTU *pmem,
	lw_u32 info[4]
)
{
	OSlwToolDListNodeSTU *pdl;
	OSlwToolNNLayerFullConSTU **pfcl;

	LwMatColType *shape_list1, *shape_list2;
	LwMatColType *pad_list;
	LwMatColType *ext_list;

	pdl = pmem->Calloc(pmem, sizeof(OSlwToolDListNodeSTU));

	shape_list1 = pmem->Calloc(pmem, 4 * sizeof(LwMatColType));
	shape_list2 = pmem->Calloc(pmem, 4 * sizeof(LwMatColType));

	pad_list = pmem->Calloc(pmem, 4 * sizeof(LwMatColType));
	ext_list = pmem->Calloc(pmem, sizeof(LwMatColType) * 2);

	shape_list1[0] = 1;
	shape_list1[1] = inz;
	shape_list1[2] = iny;
	shape_list1[3] = inx;

	ext_list[0] = 2 * move_l;
	ext_list[1] = 2 * move_l;

	pad_list[1] = pad_list[0] = kernx / 2;
	pad_list[2] = pad_list[3] = kernx / 2;

	pfcl = pmem->Calloc(pmem, sizeof(OSlwToolNNLayerFullConSTU *) * 5);

	pfcl[0] = OSlwToolNNLayerExtendNew(
		pin, pout, 4,
		shape_list1, ext_list,
		1,
		OSlwToolNNExtend_Nearest,
		pmem,
		info
	);

	shape_list2[0] = 1;
	shape_list2[1] = info[2];
	shape_list2[2] = info[1];
	shape_list2[3] = info[0];

	pfcl[1] = OSlwToolNNLayerPadNew(
		pout, pin,
		4,
		shape_list2, pad_list,
		1,
		OSlwToolNNPad_Constant,
		pmem,
		info
	);

	pfcl[2] = OSlwToolNNLayerConvNew(
		pin, pout,
		info[0], info[1], info[2],
		kernx, kerny, kernc,
		1, OSlwToolMatrixConvMethod_Valid,
		1,
		pmem,
		info
	);

	OSlwToolNNLayerConvSetIm2Col(
		(void *)(pfcl[2]),
		sizeof(fast_cnn_temp),
		fast_cnn_temp
	);

	pfcl[2]->Weight.a = we;

	pfcl[3] = OSlwToolNNLayerINormNew(
		pout, pin,
		NULL, NULL,
		info[0], info[1], info[2],
		1,
		pmem
	);

	pfcl[3]->Weight.a = gamma;
	pfcl[3]->Bias.a = beta;

	pfcl[4] = OSlwToolNNLayerActFunNew(
		pin, pout,
		info[3],
		1,
		pmem,
		pACT, 0
	);


	OSlwToolBPnnLayerAppend(pnn, pdl, 5, pfcl);
	return pfcl;

}


ParaType delt_temp;

int read_img(char *namestr,ParaType *p1,lw_u32 *ph,lw_u32 *pw)
{
	FILE *fid; 
	int h, w, i;
	int temp;

	fid = fopen(namestr, "r");

	if (fid==NULL)
	{

		printf("READ ERROR");
		exit(1);
	}

	fscanf(fid, "%d\n", &h);
	fscanf(fid, "%d\n", &w);

	printf("%d * %d\n", h, w);

	*ph = h;
	*pw = w;

	for ( i = 0; i < h*w; i++)
	{
		fscanf(fid,"%d\n", &temp);
		*p1++ = ((ParaType)temp - 123.68f);
	}

	for (i = 0; i < h*w; i++)
	{
		fscanf(fid, "%d\n", &temp);
		*p1++ = ((ParaType)temp - 116.78f);
	}

	for (i = 0; i < h*w; i++)
	{
		fscanf(fid, "%d\n", &temp);
		*p1++ = ((ParaType)temp - 103.94f);
	}

	fclose(fid);


}


void Pic2Csv(ParaType *p, lw_u32 row, lw_u32 col, char *name)
{
	lw_u32  i, j;
	FILE *file;

	file = fopen(name, "w+");
	fprintf(file, "%d\n", row);
	fprintf(file, "%d\n", col);
	for (i = 0; i < row*col*3; i++)
	{
			fprintf(file, "%f\n", *p);
			p++;
	}

	fclose(file);
}

int RunStyleTrans(ParaType *imag, lw_u32 h, lw_u32 w,char *namestr)
{

	lw_u32 info[4];
	lw_u32 i, count = 0;

	ParaType *p, *q;

	OSLW_MEM_INIT(mem);

	in_shape_list[2] = h;
	in_shape_list[3] = w;

	OSlwToolBPnnInit(&st_nn, 1);

	OSlwToolBPnnPadAppend(
		&st_nn,
		4,
		in_shape_list, in_pad_list,
		ch1data, pp0,
		OSlwToolNNPad_Constant, pmem,
		info
	);

	StyleTransConvAppend(
		&st_nn,
		info[0], info[1], info[2],
		9, 9, 32,
		1,
		pp0, pp2,
		conv1_conv_weight, conv1_IN_gamma, conv1_IN_beta,
		LwReLU, pmem, info
	);

	StyleTransConvAppend(
		&st_nn,
		info[0], info[1], info[2],
		3, 3, 64,
		2,
		pp2, pp0,
		conv2_conv_weight, conv2_IN_gamma, conv2_IN_beta,
		LwReLU, pmem, info
	);

	StyleTransConvAppend(
		&st_nn,
		info[0], info[1], info[2],
		3, 3, 128,
		2,
		pp0, pp2,
		conv3_conv_weight, conv3_IN_gamma, conv3_IN_beta,
		LwReLU, pmem, info
	);


	StyleTransResAppend(
		&st_nn,
		info[0], info[1], info[2],
		3, 3, 128,
		pp2, pp0, pp1,
		res1_residual_conv_weight, res1_residual_conv_1_weight,
		pmem, info
	);

	StyleTransResAppend(
		&st_nn,
		info[0], info[1], info[2],
		3, 3, 128,
		pp1, pp2, pp0,
		res2_residual_conv_weight, res2_residual_conv_1_weight,
		pmem, info
	);

	StyleTransResAppend(
		&st_nn,
		info[0], info[1], info[2],
		3, 3, 128,
		pp0, pp1, pp2,
		res3_residual_conv_weight, res3_residual_conv_1_weight,
		pmem, info
	);

	StyleTransResAppend(
		&st_nn,
		info[0], info[1], info[2],
		3, 3, 128,
		pp2, pp0, pp1,
		res4_residual_conv_weight, res4_residual_conv_1_weight,
		pmem, info
	);

	StyleTransResAppend(
		&st_nn,
		info[0], info[1], info[2],
		3, 3, 128,
		pp1, pp2, pp0,
		res5_residual_conv_weight, res5_residual_conv_1_weight,
		pmem, info
	);

	StyleTransExtConvAppend(
		&st_nn,
		info[0], info[1], info[2],
		3, 3, 64,
		1,
		pp0, pp2,
		deconv1_conv_transpose_conv_weight, deconv1_IN_gamma, deconv1_IN_beta,
		LwReLU, pmem, info
	);

	StyleTransExtConvAppend(
		&st_nn,
		info[0], info[1], info[2],
		3, 3, 32,
		1,
		pp2, pp0,
		deconv2_conv_transpose_conv_weight, deconv2_IN_gamma, deconv2_IN_beta,
		LwReLU, pmem, info
	);

	StyleTransConvAppend(
		&st_nn,
		info[0], info[1], info[2],
		9, 9, 3,
		1,
		pp0, pp2,
		deconv3_conv_weight, deconv3_IN_gamma, deconv3_IN_beta,
		LwTanh, pmem, info
	);


	OSlwToolBPnnAllDataInit(&st_nn, pmem);

	printf("out=[%d,%d,%d]\n", info[0], info[1], info[2]);

	printf("Running...\n");

	OSlwToolBPnnRun(&st_nn, NULL);

	Pic2Csv(pp2, info[0], info[1], namestr);

	printf("Complete\n");

	return count;

}



int main() 
{
	
	lw_u32 h, w;

	read_img("res.csv", pp2, &h, &w);
	RunStyleTrans(pp2, h, w, "style_out.csv");

	return 0;

}
