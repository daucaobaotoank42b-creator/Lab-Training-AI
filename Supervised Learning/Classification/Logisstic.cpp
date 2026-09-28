#include <stdio.h>
#include <math.h>  
#include <time.h> 
#include <stdlib.h> 

struct vector{
	double a[3]; 
}; 

typedef struct vector vector; 

double sigmoid(double x){
	return 1 / (1 + exp(-x)); 
} 

double ma(double x[], int N){            
	double max = x[0];
	for(int i = 0; i < N; i++){
		if(x[i] >= max) {
			max = x[i]; 
		}
	} 
	return max; 
} 

double mi(double x[], int N){        
	double min = x[0];
	for(int i = 0; i < N; i++){
		if(x[i] <= min) {
			min = x[i]; 
		}
	} 
	return min; 
}

double cost(vector w, vector x[], int y[], int N){
	double z[N];
	double k = 0; 
	for(int i = 0; i < N; i++) {
		z[i] = sigmoid(w.a[0] * x[i].a[0] + w.a[1] * x[i].a[1] + w.a[2]); 
	}
	for(int i = 0; i < N; i++){
		k -= y[i] * log(z[i] + 1e-15) + (1 - y[i]) * log(1 - z[i] + 1e-15); 
	} 
	return k / N; 
} 

vector gradient(vector w, vector x[], int y[], int N){
	vector k;
	k.a[0] = k.a[1] = k.a[2] = 0; 
	for(int i = 0; i < N; i++){
		for(int j = 0; j < 3; j++){
			k.a[j] += (sigmoid(w.a[0] * x[i].a[0] + w.a[1] * x[i].a[1] + w.a[2]) - y[i]) * x[i].a[j] / N; 
		} 
	} 
	return k; 
} 

vector myGD(vector w[], vector w0, double L_R, vector x[], int y[], int N){
	w[0] = w0;
	int s = 0; 
	for(int i = 0; i < 9999; i++){
		vector grad = gradient(w[i], x, y, N); 
		for(int j = 0; j < 3; j++){
			w[i + 1].a[j] = w[i].a[j] - L_R * grad.a[j]; 
		} 
		s = i; 
	} 
	return w[s + 1]; 
}

int predict(vector w[], vector r, vector w0, double L_R, vector x[], int y[], int N){
	vector q = myGD(w, w0, L_R, x, y, N);	
	if(q.a[0] * r.a[0] + q.a[1] * r.a[1] + q.a[2] >= 0){
		return 1; 
	} 
	else{
		return 0; 
	} 
} 

int main(){
	int N;
	scanf("%d", &N); 
	vector x[N];
	int y[N]; 
	for(int i = 0; i < N; i++){
		x[i].a[2] = 1; 
	} 
	for(int i = 0; i < N; i++){
		scanf("%lf %lf %d", &x[i].a[0], &x[i].a[1], &y[i]); 
	} 
	double a0[N], a1[N]; 
	for(int i = 0; i < N; i++){
		a0[i] = x[i].a[0];
		a1[i] = x[i].a[1]; 
	} 
	double max0 = ma(a0, N), max1 = ma(a1, N); 
	double min0 = mi(a0, N), min1 = mi(a1, N); 
	for(int i = 0; i < N; i++){
		x[i].a[0] = (x[i].a[0] - min0) / (max0 - min0);
		x[i].a[1] = (x[i].a[1] - min1) / (max1 - min1);
	} 
	vector w0;
	srand(time(NULL)); 
	for(int i = 0; i < 3; i++){
		w0.a[i] = (double)(rand() % 5); 
	} 
	vector w[10000]; 
	myGD(w, w0, 0.01, x, y, N); 
	double loss[10000]; 
	for(int i = 0; i < 10000; i++){
		loss[i] = cost(w[i], x, y, N); 
	} 
	FILE *f;
	f = fopen("loss_2feature.txt", "w"); 
	for(int i = 0; i < 10000; i++){
		fprintf(f, "%lf\n", loss[i]);   
	}
	fclose(f); 
//	vector s = myGD(w, w0, 0.01, x, y, N); 
//	vector ori;
//	ori.a[0] = s.a[0] / (max0 - min0);
//	ori.a[1] = s.a[1] / (max1 - min1);
//	ori.a[2] = s.a[2] - (s.a[0] * min0) / (max0 - min0) - (s.a[1] * min1) / (max1 - min1); 
//	printf("%lf %lf %lf\n", ori.a[0], ori.a[1], ori.a[2]); 
//	printf("%lf %lf\n", -ori.a[0] / ori.a[1], -ori.a[2] / ori.a[1]); 
	return 0; 
} 
