#include <stdio.h>
#include <math.h>  
#include <time.h> 
#include <stdlib.h> 

struct vector{
	double a[2]; 
}; 

typedef struct vector vector; 

double sigmoid(double x){
	return 1 / (1 + exp(-x)); 
} 

double BCELoss(vector w, vector x[], double y[], int N){
	double z[N];
	double k = 0; 
	for(int i = 0; i < N; i++) {
		z[i] = sigmoid(w.a[0] * x[i].a[0] + w.a[1]); 
	}
	for(int i = 0; i < N; i++){
		k -= y[i] * log(z[i]) + (1 - y[i]) * log(1 - z[i]); 
	} 
	return k / N; 
} 

vector gradient(vector w, vector x[], double y[], int N){
	vector k;
	k.a[0] = k.a[1] = 0; 
	for(int i = 0; i < N; i++){
		k.a[0] += (sigmoid(w.a[0] * x[i].a[0] + w.a[1]) - y[i]) * x[i].a[0] / N; 
		k.a[1] += (sigmoid(w.a[0] * x[i].a[0] + w.a[1]) - y[i]) / N;
	} 
	return k; 
} 

vector myGD(vector w[], vector w0, double L_R, vector x[], double y[], int N){
	w[0] = w0;
	int s = 0; 
	for(int i = 0; i < 5999; i++){
		vector grad = gradient(w[i], x, y, N); 
		w[i + 1].a[0] = w[i].a[0] - L_R * grad.a[0]; 
		w[i + 1].a[1] = w[i].a[1] - L_R * grad.a[1]; 
		s = i; 
	} 
	return w[s + 1]; 
}

int predict(vector w[], vector r, vector w0, double L_R, vector x[], double y[], int N){
	vector q = myGD(w, w0, L_R, x, y, N);	
	if(q.a[0] * r.a[0] + q.a[1] >= 0){
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
	double y[N]; 
	for(int i = 0; i < N; i++){
		x[i].a[1] = 1; 
	} 
	for(int i = 0; i < N; i++){
		scanf("%lf %lf", &x[i].a[0], &y[i]); 
	} 
	vector w0;
	srand(time(NULL)); 
	w0.a[0] = (double)(rand() % 5);
	w0.a[1] = (double)(rand() % 5); 
	vector w[6000]; 
	myGD(w, w0, 0.05, x, y, N); 
	double loss[6000]; 
	for(int i = 0; i < 6000; i++){
		loss[i] = BCELoss(w[i], x, y, N); 
	} 
	FILE *f;
	f = fopen("loss.txt", "w"); 
	for(int i = 0; i < 6000; i++){
		fprintf(f, "%lf\n", loss[i]);   
	}
	fclose(f); 


	return 0; 
} 
