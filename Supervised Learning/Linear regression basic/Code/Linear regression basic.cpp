#include <stdio.h> 

struct vector{
	double x;
	double y; 
}; 

typedef struct vector vector;

double square(double x){    //Ham tinh mu 2 
	return x * x; 
} 

double abs(double x){   //Ham tinh gia tri tuyet doi     
	if(x >= 0){
		return x; 
	} 
	return - x; 
} 

double cost_MSE(vector w, double x[], double y[], int N){   //Ham cost theo MSE 
	double L = 0;
	for(int i = 0; i < N; i++){
		L += square(w.x * x[i] + w.y - y[i]) / (2 * N); 
	}  
	return L; 
} 

double cost_MAE(vector w, double x[], double y[], int N){   //Ham cost theo MAE 
	double L = 0;
	for(int i = 0; i < N; i++){
		L += abs(w.x * x[i] + w.y - y[i]) / N; 
	}  
	return L; 
} 

vector grad_MSE(vector w, double x[], double y[], int N){   //Ham gradient theo MSE 
	vector p;
	p.x = 0, p.y = 0; 
	for(int i = 0; i < N; i++){
		p.x += x[i] * (w.x * x[i] + w.y - y[i]) / N;  //Tinh gradient theo w 
		p.y += (w.x * x[i] + w.y - y[i]) / N;         //Tinh gradient theo b 
	} 
	return p; 
}  

vector grad_MAE(vector w, double x[], double y[], int N){   //Ham greadient theo MAE  
	vector p;
	p.x = 0, p.y = 0; 
	for(int i = 0; i < N; i++){
		p.x += x[i] * (w.x * x[i] + w.y - y[i]) / (N * abs(w.x * x[i] + w.y - y[i]));  //Tinh gradient theo w 
		p.y += (w.x * x[i] + w.y - y[i]) / (N * abs(w.x * x[i] + w.y - y[i]));         //Tinh gradient theo b 
	}  
	return p; 
}                                               

vector myGD_MSE(vector w[], vector w0, double L_R, double x[], double y[], int N){   //Ham cap nhat tham so theo MSE 
	int s = 0; 
	w[0] = w0;; 
	for(int i = 0; i < 999; i++){
		vector tmp = grad_MSE(w[i], x, y, N); 
		w[i + 1].x = w[i].x - L_R * tmp.x;         //Cap nhat w 
		w[i + 1].y = w[i].y - L_R * tmp.y;         //Cap nhat b 
		s = i; 
	} 
	return w[s + 1]; 
} 

vector myGD_MAE(vector w[], vector w0, double L_R, double x[], double y[], int N){   //Ham cap nhat tham so theo MAE 
	int s = 0; 
	w[0] = w0;; 
	for(int i = 0; i < 999; i++){
		vector tmp = grad_MAE(w[i], x, y, N); 
		w[i + 1].x = w[i].x - L_R * tmp.x;          //Cap nhat w 
		w[i + 1].y = w[i].y - L_R * tmp.y;          //Cap nhat b 
		s = i; 
	} 
	return w[s + 1]; 
} 

//Ham du doan yhat theo MSE 
double predict_MSE(vector w[], double q, vector w0, double eta, double x[], double y[], int N){   
	vector yhat = myGD_MSE(w, w0, eta, x, y, N);
	return yhat.x * q + yhat.y;   
} 

//Ham du doan yhat theo MAE
double predict_MAE(vector w[], double q, vector w0, double eta, double x[], double y[], int N){    
	vector yhat = myGD_MAE(w, w0, eta, x, y, N);
	return yhat.x * q + yhat.y;   
} 

int main(){
	int N; 
	scanf("%d", &N);               //Nhap so luong du lieu training 
	double x[N], y[N];               
	for(int i = 0; i < N; i++){     
		scanf("%lf", &x[i]); 
	}
	for(int i = 0; i < N; i++){
		scanf("%lf", &y[i]); 
	}  
	vector w0;
	scanf("%lf %lf", &w0.x, &w0.y);                        //Nhap diem xuat phat 
	vector w_mse[1000]; 
	vector k1 = myGD_MSE(w_mse, w0, 0.05, x, y, N);
	printf("MyGD_MSE : %lf %lf\n", k1.x, k1.y); 
	printf("\n"); 
	vector w_mae[1000]; 
	vector k2 = myGD_MAE(w_mae, w0, 0.05, x, y, N);
	printf("MyGD_MAE : %lf %lf\n\n", k2.x, k2.y);
	FILE *f;
	f = fopen("so_lieu_sau_training_mse.txt", "w");        //Tao file text luu lich su training mse 
	for(int i = 0; i < 1000; i++){
		fprintf(f, "%lf %lf\n", w_mse[i].x, w_mse[i].y);   //Luu lich su training mse 
	}
	fclose(f);
	f = fopen("so_lieu_sau_training_mae.txt", "w");        //Tao file text luu lich su training mae 
	for(int i = 0; i < 1000; i++){
		fprintf(f, "%lf %lf\n", w_mae[i].x, w_mae[i].y);   //Luu lich su training mae
	} 
	fclose(f); 
} 

