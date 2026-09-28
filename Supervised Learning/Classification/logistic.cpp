#include <stdio.h> 
#include <math.h>
#include <stdlib.h>
#include <time.h> 

struct vector {
	double x;
	double y;
};

typedef struct vector vector;

double sigmoid(double x){
	return 1 / (1 + exp(-x)); 
}

double cost_logistic(vector w, vector k[], double y[], int N) {   //Binary Cross Entropy Loss 
	double L = 0;
	for (int i = 0; i < N; i++) {
		L += (- y[i] * log(sigmoid(w.x * k[i].x + w.y)) - (1 - y[i]) * log(1 - sigmoid(w.x * k[i].x + w.y))) / N;
	}
	return L;
}

vector grad(vector w, vector k[], double y[], int N) {   //Ham gradient theo logisstic 
	vector p;
	p.x = p.y = 0;
	for (int i = 0; i < N; i++) {
		p.x += (sigmoid(w.x * k[i].x + w.y) - y[i]) * k[i].x / N;         //Tinh gradient theo w 
		p.y += (sigmoid(w.x * k[i].x + w.y) - y[i]) * k[i].y / N;         //Tinh gradient theo b 
	}
	return p;
}

vector myGD(vector w[], vector w0, double L_R, vector x[], double y[], int N) {   //Ham cap nhat tham so theo MSE 
	int s = 0;
	w[0] = w0;
	for (int i = 0; i < 199999; i++) {
		vector tmp = grad(w[i], x, y, N);
		w[i + 1].x = w[i].x - L_R * tmp.x;         //Cap nhat w 
		w[i + 1].y = w[i].y - L_R * tmp.y;         //Cap nhat b 
		s = i;
	}
	return w[s + 1];
}

//Ham du doan yhat theo MSE 
int predict(vector w[], vector q, vector w0, double eta, vector x[], double y[], int N) {
	vector yhat = myGD(w, w0, eta, x, y, N);
	if(yhat.x * q.x + yhat.y >= 0){
		return 1; 
	} 
	else {
		return 0; 
	} 
}

int main() {
	int N;
	scanf("%d", &N);               //Nhap so luong du lieu training 
	vector k[N];
	double y[N];
	for(int i = 0; i < N; i++){
		k[i].y = 1; 
	} 
	for (int i = 0; i < N; i++) {
		scanf("%lf %lf", &k[i].x, &y[i]);
	}
	vector w0;
	scanf("%lf %lf", &w0.x, &w0.y);                        //Nhap diem xuat phat 
	vector BCE[200000];
	vector t = myGD(BCE, w0, 0.005, k, y, N);
	printf("MyGD : %lf %lf\n", t.x, t.y);
	FILE* f;
//	f = fopen("so_lieu_sau_training.txt", "w");        //Tao file text luu lich su training
//	for (int i = 0; i < 10000; i++) {
//		fprintf(f, "%lf %lf\n", BCE[i].x, BCE[i].y);   //Luu lich su training
//	}
//	fclose(f);
	vector u;
	u.y = 1;
	scanf("%lf", &u.x); 
	printf("%d", predict(BCE, u, w0, 0.005, k, y, N)); 
	return 0; 
}

