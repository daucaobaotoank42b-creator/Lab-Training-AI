#include <stdio.h>
#include <stdlib.h> 
#include <time.h> 

struct vector {                      //Vector hoa 
	double a[7]; 
}; 

typedef struct vector vector; 


double sq(double x){                 //Ham tinh binh phuong 
	return x * x; 
}

double abs(double x){                    //Ham tinh gia tri tuyet doi 
	if (x >= 0){
		return x; 
	} 
	else{
		return -x; 
	} 
} 

double ma(double x[], int N){            //Ham tinh max cua mot mang 
	double max = x[0];
	for(int i = 0; i < N; i++){
		if(x[i] >= max) {
			max = x[i]; 
		}
	} 
	return max; 
} 

double mi(double x[], int N){            //Tinh min cua mot mang 
	double min = x[0];
	for(int i = 0; i < N; i++){
		if(x[i] <= min) {
			min = x[i]; 
		}
	} 
	return min; 
}

vector kpv(double k, vector x){  //Ham tinh tich cua 1 so thuc va 1 vector, tra ve 1 vector 
	vector kx;
	for(int i = 0; i < 7; i++){
		kx.a[i] = k * x.a[i]; 
	} 
	return kx; 
} 

vector vpv(vector x, vector y){     //Ham tinh tong cua 2 vector, tra ve 1 vector 
	vector z;
	for(int i = 0; i < 7; i++){
		z.a[i] = x.a[i] + y.a[i]; 
	} 
	return z; 
} 

vector vmv(vector x, vector y){     //Ham tinh hieu cua 2 vector, tra ve 1 vector 
	vector z;
	for(int i = 0; i < 7; i++){
		z.a[i] = x.a[i] - y.a[i]; 
	} 
	return z; 
} 

double dp(vector x, vector y){    //Ham tinh tich vo huong cua 2 vector, tra ve 1 so thuc 
	double tvh = 0;
	for(int i = 0; i < 7; i++){
		tvh += x.a[i] * y.a[i]; 
	} 
	return tvh; 
}

double cost_MSE(vector w, vector x[], double y[], int N){                     //Ham mat mat cua MSE 
	double J = 0;
	for(int i = 0; i < N; i++){
		J += sq(dp(w, x[i]) - y[i]) / (2 * N); 
	} 
	return J; 
} 

double cost_MAE(vector w, vector x[], double y[], int N){                     //ham mat mat cua MAE 
	double J = 0;
	for(int i = 0; i < N; i++){
		J += abs(dp(w, x[i]) - y[i]) / N; 
	} 
	return J; 
}

vector grad_MSE(vector w, vector x[], double y[], int N){                 //Dao ham tai w cua MSE 
	vector G;
	for(int i = 0; i < 7; i++){
		G.a[i] = 0; 
	} 
	for(int i = 0; i < N; i++){
		G = vpv(G, kpv((dp(w, x[i]) - y[i]) / N, x[i]));  
	} 
	return G; 
} 

vector grad_MAE(vector w, vector x[], double y[], int N){                 //Dao ham tai w cua MAE 
	vector G;
	for(int i = 0; i < 7; i++){
		G.a[i] = 0; 
	}
	for(int i = 0; i < N; i++){
		double sign = 0; 
		if (dp(w, x[i]) - y[i] > 1e-9){
			sign = 1.0;	
		} 
		else if (dp(w, x[i]) - y[i] < -1e-9){
			sign = -1.0; 
		}  
		else {
			sign = 0; 
		} 
		G = vpv(G, (kpv(sign / N, x[i]))); 
	} 
	return G; 
} 

vector GD_MSE(vector w0, double LR, vector x[], double y[], int N){     //Ham cap nhat tham so theo MSE 
	vector Q;
	for(int i = 0; i < 7; i++){
		Q.a[i] = w0.a[i]; 
	} 
	for(int i = 0; i < 20000; i++){
		Q = vmv(Q, kpv(LR, grad_MSE(Q, x, y, N)));        
	} 
	return Q; 
} 

vector GD_MAE(vector w0, double LR, vector x[], double y[], int N){     //Ham cap nhat tham so theo MAE 
	vector Q;
	for(int i = 0; i < 7; i++){
		Q.a[i] = w0.a[i]; 
	} 
	for(int i = 0; i < 20000; i++){
		Q = vmv(Q, kpv(LR, grad_MAE(Q, x, y, N))); 
	} 
	return Q; 
} 

double predict_MSE(vector x0, vector w0, double LR, vector x[], double y[], int N){      //Ham du doan theo MSE 
	return dp(x0, GD_MSE(w0, LR, x, y, N)); 
} 

double predict_MAE(vector x0, vector w0, double LR, vector x[], double y[], int N){      //Ham du doan theo MAE 
	return dp(x0, GD_MAE(w0, LR, x, y, N)); 
} 

int main(){
	int N;
	scanf("%d", &N);
	vector x[N];
	double y[N]; 
	for(int i = 0; i < N; i++){
		x[i].a[6] = 1; 
	} 
	for(int i = 0; i < N; i++){
		scanf("%lf %lf %lf %lf %lf %lf %lf", &x[i].a[0], &x[i].a[1], &x[i].a[2], &x[i].a[3], &x[i].a[4], &x[i].a[5], &y[i]); 
	} 
	double Aka[N][6];
	double a0[N], a1[N], a2[N], a3[N], a4[N], a5[N]; 
	for(int i = 0; i < N; i++){
		a0[i] = x[i].a[0]; 
	} 
	for(int i = 0; i < N; i++){
		a1[i] = x[i].a[1]; 
	} 
	for(int i = 0; i < N; i++){
		a2[i] = x[i].a[2]; 
	} 
	for(int i = 0; i < N; i++){
		a3[i] = x[i].a[3]; 
	} 
	for(int i = 0; i < N; i++){
		a4[i] = x[i].a[4]; 
	} 
	for(int i = 0; i < N; i++){
		a5[i] = x[i].a[5]; 
	} 
	double max[6], min[6];
	max[0] = ma(a0, N), max[1] = ma(a1, N), max[2] = ma(a2, N), max[3] = ma(a3, N), max[4] = ma(a4, N), max[5] = ma(a5, N);
	min[0] = mi(a0, N), min[1] = mi(a1, N), min[2] = mi(a2, N), min[3] = mi(a3, N), min[4] = mi(a4, N), min[5] = mi(a5, N);
	for(int i = 0; i < N; i++){
		for(int j = 0; j < 6; j++){
			Aka[i][j] = (x[i].a[j] - min[j]) / (max[j] - min[j]); 
		} 
	} 
	for(int i = 0; i < N; i++){
		for(int j = 0; j < 6; j++){
			x[i].a[j] = Aka[i][j]; 
		} 
	}
	srand(time(NULL));
	vector w0;
	for(int i = 0; i < 7; i++){
		w0.a[i] = (double)((rand() % 9) - 3); 
	} 
	vector mse = GD_MSE(w0, 0.00015, x, y, N);
	vector mae = GD_MAE(w0, 0.0018, x, y, N); 
	for(int i = 0; i < 7; i++){
		printf("%lf ", mse.a[i]);
	} 
	printf("\n");
	for(int i = 0; i < 7; i++){
		printf("%lf ", mae.a[i]); 
	} 
	printf("\n"); 
	printf("Lost MSE: %lf\n", cost_MSE(mse, x, y, N));
	printf("Lost MAE: %lf\n", cost_MAE(mae, x, y, N));
	vector r; 
	for(int i = 0; i < 6; i++){
		scanf("%lf", &r.a[i]); 
	} 
	r.a[6] = 1; 
	for(int i = 0; i < 6; i++){
		r.a[i] = (r.a[i] - min[i]) / (max[i] - min[i]);
	} 
    printf("%lf\n", dp(r, mse));
	printf("%lf\n", dp(r, mae));
	return 0; 
} 



