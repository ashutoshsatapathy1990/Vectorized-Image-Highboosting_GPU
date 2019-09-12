//------------------------------------------------------------------------------------------------------|
// IMAGE HIGHBOOSTING IN FREQUENCY DOMAIN USING DISCRETE FOURIER TRANSFORMATION AND IDEAL FILTER.	|
// IMAGE HIGHBOOSTING IN FREQUENCY DOMAIN USING DISCRETE FOURIER TRANSFORMATION AND GAUSSIAN FILTER.	|
// IMAGE HIGHBOOSTING IN FREQUENCY DOMAIN USING DISCRETE FOURIER TRANSFORMSTION AND BUTTERWORTH FILTER. |
// IMAGE HIGHBOOSTING IN FREQUENCY DOMAIN USING DISCRETE FOURIER TRANSFORMSTION AND LoG FILTER.		|
// Image Highboosting.cpp : Defines the entry point for the console application.			|
//------------------------------------------------------------------------------------------------------|

//++++++++++++++++++++++++++++++++++ START HEADER FILES +++++++++++++++++++++++++++++++++++++++++++++
// Include The Necesssary Header Files
// [Both In std. and non std. path]
#include "stdafx.h"
#include<stdio.h>
#include<conio.h>
#include<string.h>
#include<stdlib.h>
#include<complex>
#ifdef __APPLE__
#include<OpenCL\cl.h>
#else
#include<CL\cl.h>
#endif
#include<opencv\cv.h>
#include<opencv\highgui.h>
using namespace std;
using namespace cv;
//++++++++++++++++++++++++++++++++++ END HEADER FILES +++++++++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++ RGB TO RGBA KERNEL ++++++++++++++++++++++++++++++++++++++++++++++
// OpenCL RGB to RGBA Kernel Which Is Run For Every Work Items Created.
const char* RGBA_Kernel =
"#pragma OPENCL EXTENSION cl_khr_fp32 : enable								\n" \
"__constant sampler_t image_sampler = CLK_NORMALIZED_COORDS_FALSE |					\n" \
"							CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;	\n" \
"__kernel												\n" \
"void RGBA_Kernel(__global uchar* data,									\n" \
"					__write_only image2d_t iimage,					\n" \
"					int width)							\n" \
"{													\n" \
"	int globalID = get_global_id(0);								\n" \
"	int u = globalID / width;									\n" \
"	int v = globalID % width;									\n" \
"	float4 ipixelValue;										\n" \
"	ipixelValue.x = (float)data[3 * globalID];							\n" \
"	ipixelValue.y = (float)data[3 * globalID + 1];							\n" \
"	ipixelValue.z = (float)data[3 * globalID + 2];							\n" \
"	ipixelValue.w = 255.0;										\n" \
"	write_imagef(iimage, (int2)(v, u), ipixelValue);						\n" \
"}													\n" \
"\n";
//+++++++++++++++++++++++++++++++++ END RGB To RGBA KERNEL ++++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++ HORIZONTAL DFT KERNEL +++++++++++++++++++++++++++++++++++++++++++
// OpenCL Horizontal Row Wise DFT Kernel Which Is Run For Every Work Item Created.
const char* HDFT_Kernel =
"#pragma OPENCL EXTENSION cl_khr_fp32 : enable								\n" \
"__constant sampler_t image_sampler = CLK_NORMALIZED_COORDS_FALSE |					\n" \
"							CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;	\n" \
"__kernel												\n" \
"void HDFT_Kernel(__read_only image2d_t iimage,								\n" \
"					__write_only image2d_t trimage,					\n" \
"					__write_only image2d_t tiimage,					\n" \
"					__local float4* SharedArrayR,					\n" \
"					__local float4* SharedArrayI,					\n" \
"					int width,							\n" \
"					float norm)							\n" \
"{													\n" \
"	int globalID = get_global_id(0);								\n" \
"	int localID = get_local_id(0);									\n" \
"	int groupID = get_group_id(0);									\n" \
"	int v = globalID % width;									\n" \
"	float param = (-2.0*v)/width;									\n" \
"	SharedArrayR[localID] = (0.0, 0.0, 0.0, 0.0);							\n" \
"	SharedArrayI[localID] = (0.0, 0.0, 0.0, 0.0);							\n" \
"	float c, s;											\n" \
"	float4 valueH;											\n" \
"	//valueH = read_imagef(iimage, image_sampler, (int2)(u, v));					\n" \
"	// Horizontal DFT Transformation								\n" \
"	for (int i = 0; i < width; i++)									\n" \
"	{												\n" \
"		valueH = read_imagef(iimage, image_sampler, (int2)(groupID, i));			\n" \
"		s = sinpi(i * param);									\n" \
"		c = cospi(i * param);									\n" \
"		SharedArrayR[localID] += valueH * c;							\n" \
"		SharedArrayI[localID] += valueH * s;							\n" \
"	}												\n" \
"	write_imagef(trimage, (int2)(groupID, localID), norm * SharedArrayR[localID]);			\n" \
"	write_imagef(tiimage, (int2)(groupID, localID), norm * SharedArrayI[localID]);			\n" \
"}																							\n" \
"\n";
//++++++++++++++++++++++++++++++++++++ END H-DFT KERNEL +++++++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++++ VERTICAL DFT KERNEL +++++++++++++++++++++++++++++++++++++++++++
// OpenCL Vertical Column Wise DFT Kernel Which Is Run For Every Work Item Created.
const char* VDFT_Kernel = 
"#pragma OPENCL EXTENSION cl_khr_fp32 : enable								\n" \
"__constant sampler_t image_sampler = CLK_NORMALIZED_COORDS_FALSE |					\n" \
"							CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;	\n" \
"__kernel												\n" \
"void VDFT_Kernel(__read_only image2d_t trimage,							\n" \
"				  __read_only image2d_t tiimage,					\n" \
"				  __write_only image2d_t orimage,					\n" \
"				  __write_only image2d_t oiimage,					\n" \
"				  __local float4* SharedArrayR,						\n" \
"				  __local float4* SharedArrayI,						\n" \
"					int width,							\n" \
"					float norm)							\n" \
"{													\n" \
"	size_t globalID = get_global_id(0);								\n" \
"	size_t localID = get_local_id(0);								\n" \
"	size_t groupID = get_group_id(0);								\n" \
"	int v = globalID % width;									\n" \
"	float param = (-2.0*v)/width;									\n" \
"	SharedArrayR[localID] = (0.0, 0.0, 0.0, 0.0);							\n" \
"	SharedArrayI[localID] = (0.0, 0.0, 0.0, 0.0);							\n" \
"	float c, s;											\n" \
"	float4 valueR, valueI;										\n" \
"	//valueR = read_imagef(trimage, image_sampler, (int2)(u, v));					\n" \
"	//valueI = read_imagef(tiimage, image_sampler, (int2)(u, v));					\n" \
"	// Horizontal DFT Transformation								\n" \
"	for (int i = 0; i < width; i++)									\n" \
"	{												\n" \
"		valueR = read_imagef(trimage, image_sampler, (int2)(i, groupID));			\n" \
"		valueI = read_imagef(tiimage, image_sampler, (int2)(i, groupID));			\n" \
"		s = sinpi(i * param);									\n" \
"		c = cospi(i * param);									\n" \
"		SharedArrayR[localID] += valueR * c - valueI * s;					\n" \
"		SharedArrayI[localID] += valueR * s + valueI * c;					\n" \
"	}												\n" \
"	write_imagef(orimage, (int2)(localID, groupID), norm * SharedArrayR[localID]);			\n" \
"	write_imagef(oiimage, (int2)(localID, groupID), norm * SharedArrayI[localID]);			\n" \
"}													\n" \
"\n";
//++++++++++++++++++++++++++++++++++++ END V-DFT KERNEL +++++++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++++ VERTICAL IDFT KERNEL ++++++++++++++++++++++++++++++++++++++++++
// OpenCL Vertical Column Wise IDFT Kernel Which Is Run For Every Work Item Created.
const char* VIDFT_Kernel =
"#pragma OPENCL EXTENSION cl_khr_fp32 : enable								\n" \
"__constant sampler_t image_sampler = CLK_NORMALIZED_COORDS_FALSE |					\n" \
"							CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;	\n" \
"__kernel												\n" \
"void VIDFT_Kernel(__write_only image2d_t trimage,							\n" \
"				   __write_only image2d_t tiimage,					\n" \
"				   __read_only image2d_t orimage,					\n" \
"				   __read_only image2d_t oiimage,					\n" \
"				   __local float4* SharedArrayR,					\n" \
"				   __local float4* SharedArrayI,					\n" \
"					int width,							\n" \
"					float norm)							\n" \
"{													\n" \
"	int globalID = get_global_id(0);								\n" \
"	int localID = get_local_id(0);									\n" \
"	int groupID = get_group_id(0);									\n" \
"	int v = globalID % width;									\n" \
"	float param = (2.0*v)/width;									\n" \
"	SharedArrayR[localID] = (0.0, 0.0, 0.0, 0.0);							\n" \
"	SharedArrayI[localID] = (0.0, 0.0, 0.0, 0.0);							\n" \
"	float c, s;											\n" \
"	float4 valueR, valueI;										\n" \
"	//valueR = read_imagef(orimage, image_sampler, (int2)(u, v));					\n" \
"	//valueI = read_imagef(oiimage, image_sampler, (int2)(u, v));					\n" \
"	// Horizontal IDFT Transformation								\n" \
"	for (int i = 0; i < width; i++)									\n" \
"	{												\n" \
"		valueR = read_imagef(orimage, image_sampler, (int2)(i, groupID));			\n" \
"		valueI = read_imagef(oiimage, image_sampler, (int2)(i, groupID));			\n" \
"		s = sinpi(i * param);									\n" \
"		c = cospi(i * param);									\n" \
"		SharedArrayR[localID] += valueR * c - valueI * s;					\n" \
"		SharedArrayI[localID] += valueR * s + valueI * c;					\n" \
"	}												\n" \
"	write_imagef(trimage, (int2)(localID, groupID), norm * SharedArrayR[localID]);			\n" \
"	write_imagef(tiimage, (int2)(localID, groupID), norm * SharedArrayI[localID]);			\n" \
"}													\n" \
"\n";
//++++++++++++++++++++++++++++++++++++ END V-IDFT KERNEL ++++++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++ HORIZONTAL DFT KERNEL +++++++++++++++++++++++++++++++++++++++++++
// OpenCL Horizontal Row Wise DFT Kernel Which Is Run For Every Work Item Created.
const char* HIDFT_Kernel =
"#pragma OPENCL EXTENSION cl_khr_fp32 : enable								\n" \
"__constant sampler_t image_sampler = CLK_NORMALIZED_COORDS_FALSE |					\n" \
"							CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;	\n" \
"__kernel												\n" \
"void HIDFT_Kernel(__write_only image2d_t iimage,							\n" \
"					__read_only image2d_t trimage,					\n" \
"					__read_only image2d_t tiimage,					\n" \
"					__local float4* SharedArrayR,					\n" \
"					__local float4* SharedArrayI,					\n" \
"					int width,							\n" \
"					float norm)							\n" \
"{													\n" \
"	int globalID = get_global_id(0);								\n" \
"	int localID = get_local_id(0);									\n" \
"	int groupID = get_group_id(0);									\n" \
"	int v = globalID % width;									\n" \
"	float param = (2.0*v)/width;									\n" \
"	SharedArrayR[localID] = (0.0, 0.0, 0.0, 0.0);							\n" \
"	SharedArrayI[localID] = (0.0, 0.0, 0.0, 0.0);							\n" \
"	float c, s;											\n" \
"	float4 valueR, valueI;										\n" \
"	// Horizontal DFT Transformation								\n" \
"	for (int i = 0; i < width; i++)									\n" \
"	{												\n" \
"		valueR = read_imagef(trimage, image_sampler, (int2)(groupID, i));			\n" \
"		valueI = read_imagef(tiimage, image_sampler, (int2)(groupID, i));			\n" \
"		s = sinpi(i * param);									\n" \
"		c = cospi(i * param);									\n" \
"		SharedArrayR[localID] += valueR * c - valueI * s;					\n" \
"		SharedArrayI[localID] += valueR * s + valueI * c;					\n" \
"	}												\n" \
"	write_imagef(iimage, (int2)(groupID, localID), norm * SharedArrayR[localID]);			\n" \
"}													\n" \
"\n";
//++++++++++++++++++++++++++++++++++++ END H-DFT KERNEL +++++++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++ RGBA TO RGB KERNEL ++++++++++++++++++++++++++++++++++++++++++++++
// OpenCL RGBA to RGB Kernel Which Is Run For Every Work Item Created.
const char* RGB_Kernel =
"#pragma OPENCL EXTENSION cl_khr_fp32 : enable								\n" \
"__constant sampler_t image_sampler = CLK_NORMALIZED_COORDS_FALSE |					\n" \
"							CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;	\n" \
"__kernel												\n" \
"void RGB_Kernel(__global uchar* data,									\n" \
"					__read_only image2d_t iimage,					\n" \
"					int width)							\n" \
"{													\n" \
"	int globalID = get_global_id(0);								\n" \
"	int u = globalID / width;									\n" \
"	int v = globalID % width;									\n" \
"	float4 ipixelValue;										\n" \
"	ipixelValue = read_imagef(iimage, image_sampler, (int2)(v, u));					\n" \
"	data[3 * globalID] = (uchar)ipixelValue.x;							\n" \
"	data[3 * globalID + 1] = (uchar)ipixelValue.y;							\n" \
"	data[3 * globalID + 2] = (uchar)ipixelValue.z;							\n" \
"}													\n" \
"\n";
//+++++++++++++++++++++++++++++++++ END RGBA To RGB KERNEL +++++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++++ IDEAL KERNEL ++++++++++++++++++++++++++++++++++++++++++++++++++
// OpenCL High Boost Ideal Filter Kernel Which Is Run For Every Work Item Created
const char *ideal_kernel =
"#define EXP 2.72											\n" \
"#pragma OPENCL EXTENSION cl_khr_fp32 : enable								\n" \
"__constant sampler_t image_sampler = CLK_NORMALIZED_COORDS_FALSE |					\n" \
"							CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;	\n" \
"__kernel												\n" \
"void ideal_kernel (__read_only image2d_t orimage,							\n" \
"					__read_only image2d_t oiimage,					\n" \
"					__write_only image2d_t trimage,					\n" \
"					__write_only image2d_t tiimage,					\n" \
"					int height,							\n" \
"					int width,							\n" \
"					int CUTOFF)							\n" \
"{													\n" \
"	// Get the index of work items									\n" \
"	uint index = get_global_id(0);									\n" \
"	int u = index / width;										\n" \
"	int v = index % width;										\n" \
"	float4 ipixelValueR, ipixelValueI;								\n" \
"	float D = pow(height/2 - abs(u - height/2), 2.0) 						\n" \
"							+ pow(width/2 - abs(v - width/2), 2.0);		\n" \
"	float H = 1.0 + ((sqrt(D) > CUTOFF)? 1.0 : 0.0);						\n" \
"	ipixelValueR = read_imagef(orimage, image_sampler, (int2)(u, v));				\n" \
"	ipixelValueI = read_imagef(oiimage, image_sampler, (int2)(u, v));				\n" \
"	write_imagef(trimage, (int2)(u, v), ipixelValueR * H);						\n" \
"	write_imagef(tiimage, (int2)(u, v), ipixelValueI * H);						\n" \
"}													\n" \
"\n";
//+++++++++++++++++++++++++++++++++ END IDEAL KERNEL ++++++++++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++++ GAUSSIAN KERNEL +++++++++++++++++++++++++++++++++++++++++++++++
// OpenCL High Boost Gaussian Filter Kernel Which Is Run For Every Work Item Created
const char *gaussian_kernel =
"#define EXP 2.72											\n" \
"#pragma OPENCL EXTENSION cl_khr_fp32 : enable								\n" \
"__constant sampler_t image_sampler = CLK_NORMALIZED_COORDS_FALSE |					\n" \
"							CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;	\n" \
"__kernel												\n" \
"void gaussian_kernel (__read_only image2d_t orimage,							\n" \
"						__read_only image2d_t oiimage,				\n" \
"						__write_only image2d_t trimage,				\n" \
"						__write_only image2d_t tiimage,				\n" \
"						int height,						\n" \
"						int width,						\n" \
"						int CUTOFF)						\n" \
"{													\n" \
"	// Get the index of work items									\n" \
"	uint index = get_global_id(0);									\n" \
"	int u = index / width;										\n" \
"	int v = index % width;										\n" \
"	float4 ipixelValueR, ipixelValueI;								\n" \
"	float D = pow(height/2 - abs(u - height/2), 2.0) 						\n" \
"							+ pow(width/2 - abs(v - width/2), 2.0);		\n" \
"	float H = 2.0 - pow(EXP, (-1.0 * D / (2.0 * pow(CUTOFF, 2.0))));				\n" \
"	ipixelValueR = read_imagef(orimage, image_sampler, (int2)(u, v));				\n" \
"	ipixelValueI = read_imagef(oiimage, image_sampler, (int2)(u, v));				\n" \
"	write_imagef(trimage, (int2)(u, v), ipixelValueR * H);						\n" \
"	write_imagef(tiimage, (int2)(u, v), ipixelValueI * H);						\n" \
"}													\n" \
"\n";
//+++++++++++++++++++++++++++++++++ END GAUSSIAN KERNEL +++++++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++++ BUTTERWORTH KERNEL ++++++++++++++++++++++++++++++++++++++++++++
// OpenCL High Boost Butterworth Filter Kernel Which Is Run For Every Work Item Created
const char *butterworth_kernel =
"#define EXP 2.72											\n" \
"#pragma OPENCL EXTENSION cl_khr_fp32 : enable								\n" \
"__constant sampler_t image_sampler = CLK_NORMALIZED_COORDS_FALSE |					\n" \
"							CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;	\n" \
"__kernel												\n" \
"void butterworth_kernel (__read_only image2d_t orimage,						\n" \
"						 __read_only image2d_t oiimage,				\n" \
"						 __write_only image2d_t trimage,			\n" \
"						 __write_only image2d_t tiimage,			\n" \
"						int height,						\n" \
"						int width,						\n" \
"						int CUTOFF,						\n" \
"						float Ord)						\n" \
"{													\n" \
"	// Get the index of work items									\n" \
"	uint index = get_global_id(0);									\n" \
"	int u = index / width;										\n" \
"	int v = index % width;										\n" \
"	float4 ipixelValueR, ipixelValueI;								\n" \
"	float D = pow(height/2 - abs(u - height/2), 2.0) 						\n" \
"					+ pow(width/2 - abs(v - width/2), 2.0);				\n" \
"	float H = 1.0 + 1.0 / (1 + pow (CUTOFF / sqrt(D), 2 * Ord));					\n" \
"	ipixelValueR = read_imagef(orimage, image_sampler, (int2)(u, v));				\n" \
"	ipixelValueI = read_imagef(oiimage, image_sampler, (int2)(u, v));				\n" \
"	write_imagef(trimage, (int2)(u, v), ipixelValueR * H);						\n" \
"	write_imagef(tiimage, (int2)(u, v), ipixelValueI * H);						\n" \
"}													\n" \
"\n";
//+++++++++++++++++++++++++++++++++ END BUTTERWORTH KERNEL ++++++++++++++++++++++++++++++++++++++++++

//++++++++++++++++++++++++++++++ LAPLACIAN OF GAUSSIAN KERNEL +++++++++++++++++++++++++++++++++++++++
// OpenCL Highboost LoG Filter Kernel Which Is Run For Every Work Item Created
const char *LoG_kernel =
"#define EXP 2.72											\n" \
"#pragma OPENCL EXTENSION cl_khr_fp32 : enable								\n" \
"__constant sampler_t image_sampler = CLK_NORMALIZED_COORDS_FALSE |					\n" \
"							CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;	\n" \
"__kernel												\n" \
"void LoG_kernel (__read_only image2d_t orimage,							\n" \
"					__read_only image2d_t oiimage,					\n" \
"					__write_only image2d_t trimage,					\n" \
"					__write_only image2d_t tiimage,					\n" \
"					int height,							\n" \
"					int width,							\n" \
"					int CUTOFF)							\n" \
"{													\n" \
"	// Get the index of work items									\n" \
"	uint index = get_global_id(0);									\n" \
"	int u = index / width;										\n" \
"	int v = index % width;										\n" \
"	float Freq = pow(CUTOFF, 2.0);									\n" \
"	float4 ipixelValueR, ipixelValueI;								\n" \
"	float D = pow(height/2 - abs(u - height/2), 2.0) 						\n" \
"						+ pow(width/2 - abs(v - width/2), 2.0);			\n" \
"	float H = 2.0 - (1.0 - D / Freq) * pow(EXP, -1.0 * D / (2.0 * Freq));				\n" \
"	ipixelValueR = read_imagef(orimage, image_sampler, (int2)(u, v));				\n" \
"	ipixelValueI = read_imagef(oiimage, image_sampler, (int2)(u, v));				\n" \
"	write_imagef(trimage, (int2)(u, v), ipixelValueR * H);						\n" \
"	write_imagef(tiimage, (int2)(u, v), ipixelValueI * H);						\n" \
"}													\n" \
"\n";
//++++++++++++++++++++++++++ END LAPLACIAN OF GAUSSIAN KERNEL +++++++++++++++++++++++++++++++++++++++

// ++++++++++++++++++++++++++++++++ + CREATE AND BUILD PROGRAM++++++++++++++++++++++++++++++++++++++++
// Create and Build The Program Using Selected High Boost Filter's OpenCL Kernel Source Code.
cl_program Highboost_Program(cl_context context, const char *name, cl_device_id* device_list, cl_int clStatus)
{
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&name, NULL, &clStatus);
	clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
	return program;
}
//++++++++++++++++++++++++++++++ END CREATE AND BUILD PROGRAM +++++++++++++++++++++++++++++++++++++++

//++++++++++++++++++++++++++++ SET RGBA/RGB KERNELS' ARGUMENTS ++++++++++++++++++++++++++++++++++++++
// Passing Arguments to RGBA and RGB Kernels Dedicated for Prior/Posterior Operations After Horizontal/Vertical DFT and IDFT.
void Convert_Kernel_Arg(cl_kernel kernel, cl_mem data, cl_mem iimage, int cols, cl_int clStatus)
{
	clStatus = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&data);
	clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&iimage);
	clStatus = clSetKernelArg(kernel, 2, sizeof(int), (void *)&cols);
}
//++++++++++++++++++++++++++ END RGB/RGBA KERNELS' ARGUMENTS +++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++ SET HDFT KERNELS' ARGUMENTS ++++++++++++++++++++++++++++++++++++++
// Passing Arguments to HDFT and HIDFT Kernels Dedicated for Horizontal DFT and IDFT.
void HDFT_Kernel_Arg(cl_kernel kernel, cl_mem iimage, cl_mem trimage, cl_mem tiimage, int cols, float norm, cl_int clStatus)
{
	clStatus = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&iimage);
	clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&trimage);
	clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&tiimage);
	clStatus = clSetKernelArg(kernel, 3, sizeof(float) * 4 * cols, NULL);
	clStatus = clSetKernelArg(kernel, 4, sizeof(float) * 4 * cols, NULL);
	clStatus = clSetKernelArg(kernel, 5, sizeof(int), (void *)&cols);
	clStatus = clSetKernelArg(kernel, 6, sizeof(float), (void *)&norm);
}
//+++++++++++++++++++++++++++++ END OPENCL HDFT KERNELS' ARGUMENTS ++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++ SET VDFT KERNELS' ARGUMENTS ++++++++++++++++++++++++++++++++++++++
// Passing Arguments to VDFT and VIDFT Kernels Dedicated for Horizontal DFT and IDFT.
void VDFT_Kernel_Arg(cl_kernel kernel, cl_mem trimage, cl_mem tiimage, cl_mem orimage, cl_mem oiimage, int cols, float norm, cl_int clStatus)
{
	clStatus = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&trimage);
	clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&tiimage);
	clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&orimage);
	clStatus = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&oiimage);
	clStatus = clSetKernelArg(kernel, 4, sizeof(float) * 4 * cols, NULL);
	clStatus = clSetKernelArg(kernel, 5, sizeof(float) * 4 * cols, NULL);
	clStatus = clSetKernelArg(kernel, 6, sizeof(int), (void *)&cols);
	clStatus = clSetKernelArg(kernel, 7, sizeof(float), (void *)&norm);
}
//+++++++++++++++++++++++++++++ END OPENCL VDFT KERNELS' ARGUMENTS ++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++ SET FILTER KERNELS' ARGUMENTS ++++++++++++++++++++++++++++++++++++++
// Passing Arguments to Filter Kernels Dedicated for Image High Boosting.
void Kernel_Arg(cl_kernel kernel, cl_mem orimage, cl_mem oiimage, cl_mem trimage, cl_mem tiimage, int rows, int cols, int cutoff, float ord, int Select, cl_int clStatus)
{
	clStatus = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&orimage);
	clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&oiimage);
	clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&trimage);
	clStatus = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&tiimage);
	clStatus = clSetKernelArg(kernel, 4, sizeof(int), (void *)&rows);
	clStatus = clSetKernelArg(kernel, 5, sizeof(int), (void *)&cols);
	clStatus = clSetKernelArg(kernel, 6, sizeof(int), (void *)&cutoff);
	if (Select == 3)
	{
		clStatus = clSetKernelArg(kernel, 7, sizeof(float), (void *)&ord);
	}
}
//+++++++++++++++++++++++++++++ END OPENCL FILTER KERNELS' ARGUMENTS ++++++++++++++++++++++++++++++++++

//++++++++++++++++++++++++++++++++++ START MAIN PROGRAM +++++++++++++++++++++++++++++++++++++++++++++
int main()
{
	// Declare Variables for CUTOFF Frequency and Order of The Filter
	// Delcare Variables for Selection of a Filter.
	int CUTOFF, Select;
	float Ord;

	// Initialize Clock Variable to compute Time Taken in millisecs
	clock_t start, end;
	float Time_Used;

	// Enter CUTOFF Frequency and Order of The Filter
	// Enter Selection of Filter [1: Ideal, 2: Gaussian, 3: Butterworth, 4: LoG]
	printf("ENTER THE CUTOFF FREQUENCY AND ORDER OF THE FILTER: \n");
	scanf("%d %f", &CUTOFF, &Ord);
	printf("ENTER YOUR CHOICE FOR IMAGE DENOISIFICATION. [1]IDEAL [2]GAUSSIAN [3]BUTTERWORTH [4]LoG :\n");
	scanf("%d", &Select);

	// Get The Platforms' Information
	cl_platform_id* platforms = NULL;
	cl_uint num_platforms;

	// Set up The Platforms
	cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);


	// Get The Device Lists and Choose The Device You Want to Run on.
	cl_device_id* device_list = NULL;
	cl_uint num_devices;
	clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	device_list = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
	clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, num_devices, device_list, NULL);

	// Create an OpenCL Context for Each Device in The Platform
	cl_context context;
	context = clCreateContext(NULL, num_devices, device_list, NULL, NULL, &clStatus);

	// Create a Command Queue for Out of Order Execution in 0th Device.
	cl_command_queue command_queue_highboost = clCreateCommandQueue(context, device_list[0], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &clStatus);

	// Get The Program's and Kernels' Information
	cl_program program_RGBA = NULL, program_RGB = NULL, program_HDFT = NULL, program_HIDFT = NULL, program_VDFT = NULL, program_VIDFT = NULL, program_highboost = NULL;
	cl_kernel kernel_RGBA = NULL, kernel_RGB = NULL, kernel_HDFT = NULL, kernel_HIDFT = NULL, kernel_VDFT = NULL, kernel_VIDFT = NULL, kernel_highboost = NULL;

	// Creation and Building of OpenCL DFT and IDFT Programs for a RGB Image .
	program_RGBA = Highboost_Program(context, RGBA_Kernel, device_list, clStatus);
	program_RGB = Highboost_Program(context, RGB_Kernel, device_list, clStatus);
	program_HDFT = Highboost_Program(context, HDFT_Kernel, device_list, clStatus);
	program_HIDFT = Highboost_Program(context, HIDFT_Kernel, device_list, clStatus);
	program_VDFT = Highboost_Program(context, VDFT_Kernel, device_list, clStatus);
	program_VIDFT = Highboost_Program(context, VIDFT_Kernel, device_list, clStatus);

	// Creation of OpenCL DFT and IDFT Kernels for Image Highboosting Operation.
	kernel_RGBA = clCreateKernel(program_RGBA, "RGBA_Kernel", &clStatus);
	kernel_RGB = clCreateKernel(program_RGB, "RGB_Kernel", &clStatus);
	kernel_HDFT = clCreateKernel(program_HDFT, "HDFT_Kernel", &clStatus);
	kernel_HIDFT = clCreateKernel(program_HIDFT, "HIDFT_Kernel", &clStatus);
	kernel_VDFT = clCreateKernel(program_VDFT, "VDFT_Kernel", &clStatus);
	kernel_VIDFT = clCreateKernel(program_VIDFT, "VIDFT_Kernel", &clStatus);

	// Read Noisy Image from The Given Path
	Mat Image = imread("Impulse.bmp", CV_LOAD_IMAGE_COLOR);

	// Check The Status of Image <Mat Variable>
	if (!Image.data)
	{
		printf("COULDN'T OPEN OR READ INPUT FILE");
		return -1;
	}

	// Display The Input Blurred Image
	namedWindow("BLURRED IMAGE", WINDOW_NORMAL);
	imshow("BLURRED IMAGE", Image);

	// Specify Image Format [Image Data Type and Its Channels].
	cl_image_format image_format;
	image_format.image_channel_data_type = CL_FLOAT;
	image_format.image_channel_order = CL_RGBA;

	// Image Descriptor to Represent Structure of An Image.
	cl_image_desc image_desc;
	image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	image_desc.image_width = Image.cols;
	image_desc.image_height = Image.rows;
	image_desc.image_depth = 1;
	image_desc.image_array_size = 1;
	image_desc.image_row_pitch = 0;
	image_desc.image_slice_pitch = 0;
	image_desc.num_mip_levels = 0;
	image_desc.num_samples = 0;
	image_desc.buffer = NULL;

	// Create Host Buffers as Image Size [H x W] to Store Image Pixel Values.
	uchar* ORI_Frame = (uchar*)malloc(sizeof(uchar) * Image.rows * Image.cols * 3);

	// Create OpenCL Device Buffers and Map to Host Buffers to Device Buffers for Kernel Execution.
	cl_mem ORI_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(uchar) * Image.rows * Image.cols * 3, ORI_Frame, &clStatus);
	cl_mem IMG_clmem = clCreateImage(context, CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &clStatus);
	cl_mem TRIMG_clmem = clCreateImage(context, CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &clStatus);
	cl_mem TIIMG_clmem = clCreateImage(context, CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &clStatus);
	cl_mem ORIMG_clmem = clCreateImage(context, CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &clStatus);
	cl_mem OIIMG_clmem = clCreateImage(context, CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &clStatus);

	// Passing Arguments to DFT and IDFT Kernels for Image Transformation Before and After Image Enhancement.
	Convert_Kernel_Arg(kernel_RGBA, ORI_clmem, IMG_clmem, Image.cols, clStatus);
	HDFT_Kernel_Arg(kernel_HDFT, IMG_clmem, TRIMG_clmem, TIIMG_clmem, Image.cols, 1.0 / sqrt(Image.cols), clStatus);
	HDFT_Kernel_Arg(kernel_HIDFT, IMG_clmem, ORIMG_clmem, OIIMG_clmem, Image.cols, 1.0 / sqrt(Image.cols), clStatus);
	VDFT_Kernel_Arg(kernel_VDFT, TRIMG_clmem, TIIMG_clmem, ORIMG_clmem, OIIMG_clmem, Image.cols, 1.0 / sqrt(Image.cols), clStatus);
	VDFT_Kernel_Arg(kernel_VIDFT, ORIMG_clmem, OIIMG_clmem, TRIMG_clmem, TIIMG_clmem, Image.cols, 1.0 / sqrt(Image.cols), clStatus);
	Convert_Kernel_Arg(kernel_RGB, ORI_clmem, IMG_clmem, Image.cols, clStatus);

	// Selection of High Boost Filter: [1]Ideal [2]Gaussian [3]Butterworth [4]LoG.
	// Creation and Building of OpenCL Programs and Kernels for Each Channels [R, G, B].
	// Passing Arguments to Kernels for Image High boosting based on Selected Filter.
	switch (Select)
	{
	case 1: printf("YOU HAVE SELECTED IDEAL FILTER \n");
		program_highboost = Highboost_Program(context, ideal_kernel, device_list, clStatus);
		kernel_highboost = clCreateKernel(program_highboost, "ideal_kernel", &clStatus);
		Kernel_Arg(kernel_highboost, ORIMG_clmem, OIIMG_clmem, TRIMG_clmem, TIIMG_clmem, Image.rows, Image.cols, CUTOFF, Ord, Select, clStatus);
		break;
	case 2: printf("YOU HAVE SELECTED GAUSSIAN FILTER \n");
		program_highboost = Highboost_Program(context, gaussian_kernel, device_list, clStatus);
		kernel_highboost = clCreateKernel(program_highboost, "gaussian_kernel", &clStatus);
		Kernel_Arg(kernel_highboost, ORIMG_clmem, OIIMG_clmem, TRIMG_clmem, TIIMG_clmem, Image.rows, Image.cols, CUTOFF, Ord, Select, clStatus);
		break;
	case 3: printf("YOU HAVE SELECTED BUTTERWORTH FILTER \n");
		program_highboost = Highboost_Program(context, butterworth_kernel, device_list, clStatus);
		kernel_highboost = clCreateKernel(program_highboost, "butterworth_kernel", &clStatus);
		Kernel_Arg(kernel_highboost, ORIMG_clmem, OIIMG_clmem, TRIMG_clmem, TIIMG_clmem, Image.rows, Image.cols, CUTOFF, Ord, Select, clStatus);
		break;
	case 4: printf("YOU HAVE SELECTED LoG FILTER \n");
		program_highboost = Highboost_Program(context, LoG_kernel, device_list, clStatus);
		kernel_highboost = clCreateKernel(program_highboost, "LoG_kernel", &clStatus);
		Kernel_Arg(kernel_highboost, ORIMG_clmem, OIIMG_clmem, TRIMG_clmem, TIIMG_clmem, Image.rows, Image.cols, CUTOFF, Ord, Select, clStatus);
		break;
	default:printf("YOU HAVE SELECTED WRONG FILTER \n");
		break;
	}

	//  Set Global Size of Index Space and Local Size of Work Group.
	size_t global = Image.rows * Image.cols, local = 256;

	// Start The Timer
	start = clock();

	// Copy Image Data to Host Buffers for Mapping to Device Buffer.
	memcpy(ORI_Frame, Image.data, Image.rows * Image.cols * sizeof(uchar) * 3);

	// Convert An Image from RGB Channels to RGBA Channels Before DFT Operation Using Image Vectorization.
	clStatus = clEnqueueNDRangeKernel(command_queue_highboost, kernel_RGBA, 1, NULL, &global, &local, 0, NULL, NULL);

	// Perform DFT Transforms [Horizontal DFT: Row wise DFT] + [Vertical DFT : Column wise DFT]
	clStatus = clEnqueueNDRangeKernel(command_queue_highboost, kernel_HDFT, 1, NULL, &global, &local, 0, NULL, NULL);
	clStatus = clEnqueueNDRangeKernel(command_queue_highboost, kernel_VDFT, 1, NULL, &global, &local, 0, NULL, NULL);

	//Execute the OpenCL Kernels for Boosting of Input Image Independetly.
	clStatus = clEnqueueNDRangeKernel(command_queue_highboost, kernel_highboost, 1, NULL, &global, &local, 0, NULL, NULL);

	// Perform DFT Transforms [Vertical IDFT : Column wise IDFT] + [Horizontal IDFT: Row wise IDFT]
	clStatus = clEnqueueNDRangeKernel(command_queue_highboost, kernel_VIDFT, 1, NULL, &global, &local, 0, NULL, NULL);
	clStatus = clEnqueueNDRangeKernel(command_queue_highboost, kernel_HIDFT, 1, NULL, &global, &local, 0, NULL, NULL);

	// Convert An Image from RGBA Channels to RGB Channels After IDFT Operation Using Image Vectorization.
	clStatus = clEnqueueNDRangeKernel(command_queue_highboost, kernel_RGB, 1, NULL, &global, &local, 0, NULL, NULL);

	//Read Data from Device Buffer to Host Buffer After Image Enhancement.
	clStatus = clEnqueueReadBuffer(command_queue_highboost, ORI_clmem, CL_TRUE, 0, Image.rows * Image.cols * sizeof(uchar) * 3, ORI_Frame, 0, NULL, NULL);

	// Copy From Host Buffers to Image Container for Displaying Purposes.
	memcpy(Image.data, ORI_Frame, 3 * Image.rows * Image.cols * sizeof(uchar));

	//Stop the Timer
	end = clock();

	// Calculate Time Taken for Image Restoration by a Low Pass Filter in millisecs.
	Time_Used = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Time Taken for Image smoothing : %.3f", Time_Used);

	// Display Final Restored Image By Low Pass Fi lter
	namedWindow("ENHANCED IMAGE", WINDOW_NORMAL);
	imshow("ENHANCED IMAGE", Image);

	// Finally Release All OpenCL Allocated Objects and Buffers [Host & Device].
	clStatus = clReleaseKernel(kernel_highboost);
	clStatus = clReleaseKernel(kernel_HDFT);
	clStatus = clReleaseKernel(kernel_HIDFT);
	clStatus = clReleaseKernel(kernel_VDFT);
	clStatus = clReleaseKernel(kernel_VIDFT);
	clStatus = clReleaseKernel(kernel_RGBA);
	clStatus = clReleaseKernel(kernel_RGB);
	clStatus = clReleaseProgram(program_highboost);
	clStatus = clReleaseProgram(program_HDFT);
	clStatus = clReleaseProgram(program_HIDFT);
	clStatus = clReleaseProgram(program_VDFT);
	clStatus = clReleaseProgram(program_VIDFT);
	clStatus = clReleaseProgram(program_RGB);
	clStatus = clReleaseProgram(program_RGBA);
	clStatus = clReleaseMemObject(ORIMG_clmem);
	clStatus = clReleaseMemObject(OIIMG_clmem);
	clStatus = clReleaseMemObject(TRIMG_clmem);
	clStatus = clReleaseMemObject(TIIMG_clmem);
	clStatus = clReleaseMemObject(IMG_clmem);
	clStatus = clReleaseMemObject(ORI_clmem);
	clStatus = clReleaseCommandQueue(command_queue_highboost);
	clStatus = clReleaseContext(context);
	free(ORI_Frame);
	Image.release();

	waitKey(0);
	return 0;
}
//++++++++++++++++++++++++++++++++++++++ END MAIN PROGRAM ++++++++++++++++++++++++++++++++++++++++++
