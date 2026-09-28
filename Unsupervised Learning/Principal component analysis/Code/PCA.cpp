#include <stdio.h>
#include <stdlib.h>
#include <math.h> 

typedef struct vector{
	double x;
	double y;
	double z; 
} vector;

double mu2(double x){
	return x * x; 
}
 
void datafile_to_array(vector a[], int n, FILE *f){
	if(f == NULL){
		printf("Khong the mo file !\n"); 
	} 
	else{
		int i = 0, j = 0; 
		double p; 
		while(fscanf(f, "%lf", &p) != -1){
			if(i % 3 == 0){
				a[j].x = p; 
			} 
			else if(i % 3 == 1){
				a[j].y = p; 
			} 
			else{
				a[j].z = p; 
				j++; 
			}
			i++;  
		} 
	} 
} 

vector vt_don_vi(vector a){
	if(a.x == 0 && a.y == 0 && a.z == 0){
		return a; 
	} 
	double do_dai = sqrt(mu2(a.x) + mu2(a.y) + mu2(a.z));
	a.x /= do_dai;
	a.y /= do_dai;
	a.z /= do_dai; 
	return a; 
} 

vector mean(vector a[], int n){
	vector m;
	m.x = 0, m.y = 0, m.z = 0; 
	for(int i = 0; i < n; i++){
		m.x += a[i].x;
		m.y += a[i].y; 
		m.z += a[i].z;
	} 
	m.x /= n;
	m.y /= n;
	m.z /= n;
	return m; 
} 

void scale(vector a[], int n){
	vector m = mean(a, n); 
	for(int i = 0; i < n; i++){
		a[i].x = a[i].x - m.x;
		a[i].y = a[i].y - m.y;  
		a[i].z = a[i].z - m.z;
	} 
} 

vector sum_square(vector a[], int n){
	vector sum;
	sum.x = 0, sum.y = 0, sum.z = 0; 
	for(int i = 0; i < n; i++){
		sum.x += mu2(a[i].x); 
		sum.y += mu2(a[i].y);
		sum.z += mu2(a[i].z); 
	} 
	sum.x /= n;
	sum.y /= n;
	sum.z /= n;
	return sum; 
} 

double sum_xy(vector a[], int n){
	double sum = 0;
	for(int i = 0; i < n; i++){
		sum += a[i].x * a[i].y; 
	} 
	return sum / n; 
} 

vector *vector_rieng(vector a[], int n){
	vector psai = sum_square(a, n);
	double xy = sum_xy(a, n);
	double delta = mu2(psai.x + psai.y) - 4 * (psai.x * psai.y - mu2(xy));
	vector tri_r;
	tri_r.x = (psai.x + psai.y + sqrt(delta)) / 2;
	tri_r.y = (psai.x + psai.y - sqrt(delta)) / 2;
	vector *vt_rieng = (vector*)malloc(2 * sizeof(vector));
	vt_rieng[0].x = - xy;
	vt_rieng[0].y = psai.x - tri_r.x;
	vt_rieng[1].x = -xy;
	vt_rieng[1].y = psai.x - tri_r.y;
	vt_rieng[0] = vt_don_vi(vt_rieng[0]);
	vt_rieng[1] = vt_don_vi(vt_rieng[1]);
	printf("%lf %lf\n", vt_rieng[0].x, vt_rieng[0].y); 
	printf("%lf %lf\n", vt_rieng[1].x, vt_rieng[1].y);
	return vt_rieng;
} 

vector doi_co_so(vector a, vector *vtr){
	vector dcs;
	dcs.x = (vtr[0].x * a.x + vtr[0].y * a.y) / (mu2(vtr[0].x) + mu2(vtr[0].y)); 
	dcs.y = (vtr[0].y * a.x - vtr[0].x * a.y) / (mu2(vtr[0].x) + mu2(vtr[0].y)); 
	return dcs; 
} 



int main(){
	vector data[500];
	FILE *f;
	f = fopen("pca_500_samples_2d.txt", "r");
	datafile_to_array(data, 500, f);
	fclose(f); 
	scale(data, 500); 
	FILE *p;
	p = fopen("pca_500_scale.txt", "w");
	for(int i = 0; i < 500; i++){
		fprintf(p, "%lf %lf\n", data[i].x, data[i].y);
	}
	fclose(p);
	vector *vtr = vector_rieng(data, 500);
	vector data_new_co_so[500];
	for(int i = 0; i < 500; i++){
		data_new_co_so[i] = doi_co_so(data[i], vtr); 
	} 
	FILE *q;
	q = fopen("pca_500_doi_co_so.txt", "w");
	for(int i = 0; i < 500; i++){
		fprintf(q, "%lf %lf\n", data_new_co_so[i].x, data_new_co_so[i].y);
	}
	fclose(q);
	
	FILE *v;
	v = fopen("pca_500_doi_co_so_1d.txt", "w");
	for(int i = 0; i < 500; i++){
		fprintf(v, "%lf 0\n", data_new_co_so[i].x);
	}
	fclose(v);
} 
