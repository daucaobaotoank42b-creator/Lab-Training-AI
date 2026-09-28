#include <stdio.h> 
#include <stdlib.h>
#include <math.h> 

struct customer{
	double c[11];                                     
};  

struct cap{
	double q[2]; 
}; 

typedef struct customer customer;
typedef struct cap cap; 

double max(double a[], int n){
	double max = a[0];
	for(int i = 1; i < n; i++){
		if(a[i] >= max){
			max = a[i]; 
		} 
	} 
	return max; 
} 

double min(double a[], int n){
	double min = a[0];
	for(int i = 1; i < n; i++){
		if(a[i] <= min){
			min = a[i]; 
		} 
	} 
	return min; 
} 

double mu2(double x){
	return x * x; 
}

double lomuto(double a[], int left, int right){
	int i = left - 1, pivot = a[right];
	for(int j = left; j < right; j++){
		if(a[j] <= pivot){
		++i;
		double tmp = a[i];
		a[i] = a[j];
		a[j] = tmp; 
		} 
	} 
	++i;
	double tmp = a[i];
	a[i] = a[right];
	a[right] = tmp;
	return i; 
} 

void quickSort(double a[], int left, int right){
	if(left < right){
		int pos = lomuto(a, left, right);
		quickSort(a, left, pos - 1);
		quickSort(a, pos + 1, right); 
	} 
} 

double partition(cap a[], int left, int right){
	int i = left - 1;
	double pivot = a[right].q[1];
	for(int j = left; j < right; j++){
		if(a[j].q[1] <= pivot){
		++i;
		cap tmp = a[i];
		a[i] = a[j];
		a[j] = tmp; 
		} 
	} 
	++i;
	cap tmp = a[i];
	a[i] = a[right];
	a[right] = tmp;
	return i; 
} 

void Sort(cap a[], int left, int right){
	if(left < right){
		int pos = partition(a, left, right);
		Sort(a, left, pos - 1);
		Sort(a, pos + 1, right); 
	} 
} 

cap maxmin_A(customer a[], int n){
	double t[n];
	for(int i = 0; i < n; i++){
		t[i] = a[i].c[1]; 
	} 
	cap r; 
	r.q[0] = max(t, n);
	r.q[1] = min(t, n); 
	return r; 
}

cap maxmin_T(customer a[], int n){
	double t[n];
	for(int i = 0; i < n; i++){
		t[i] = a[i].c[3]; 
	} 
	cap r; 
	r.q[0] = max(t, n);
	r.q[1] = min(t, n); 
	return r; 
}

cap maxmin_MC(customer a[], int n){
	double t[n];
	for(int i = 0; i < n; i++){
		t[i] = a[i].c[8]; 
	} 
	cap r; 
	r.q[0] = max(t, n);
	r.q[1] = min(t, n); 
	return r; 
}

cap maxmin_TC(customer a[], int n){
	double t[n];
	for(int i = 0; i < n; i++){
		t[i] = a[i].c[9]; 
	} 
	cap r; 
	r.q[0] = max(t, n);
	r.q[1] = min(t, n); 
	return r; 
}

void feature_scaling(customer a[], int n){
	double x1[n];
	cap r_A = maxmin_A(a, n);
	cap r_T = maxmin_T(a, n);
	cap r_MC = maxmin_MC(a, n);
	cap r_TC = maxmin_TC(a, n);
	for(int i = 0; i < n; i++){
		a[i].c[1] = (a[i].c[1] - r_A.q[1]) / (r_A.q[0] - r_A.q[1]);
		a[i].c[3] = (a[i].c[3] - r_T.q[1]) / (r_T.q[0] - r_T.q[1]);
		a[i].c[8] = (a[i].c[8] - r_MC.q[1]) / (r_MC.q[0] - r_MC.q[1]);
		a[i].c[9] = (a[i].c[9] - r_TC.q[1]) / (r_TC.q[0] - r_TC.q[1]);
	} 
}

double distanceG(customer a, customer b){
	if(a.c[2] == b.c[2]){
		return 0;
	}
	return 1;
}

double distanceCT(customer a, customer b){
	if(a.c[4] == b.c[4]){
		return 0;
	}
	return 1;
}

double distancePB(customer a, customer b){
	if(a.c[5] == b.c[5]){
		return 0;
	}
	return 1;
}

double distanceIS(customer a, customer b){
	if(a.c[6] == b.c[6]){
		return 0;
	}
	return 1;
}

double distanceTS(customer a, customer b){
	if(a.c[7] == b.c[7]){
		return 0;
	}
	return 1;
}

double distance(customer a, customer b){
	double k = mu2(a.c[1] - b.c[1]) + mu2(a.c[3] - b.c[3]) + mu2(a.c[8] - b.c[8]) + mu2(a.c[9] - b.c[9]);
	k += distanceG(a, b) + distanceCT(a, b) + distancePB(a, b) + distanceIS(a, b) + distanceTS(a, b);
	return sqrt(k);
}

int predict(customer a[], int n, customer x, int k){
	cap d[n];
	for(int i = 0; i < n; i++){
		d[i].q[0] = i;
		d[i].q[1] = distance(x, a[i]); 
	} 
	Sort(d, 0, n - 1);
	int array[k];
	for(int i = 0; i < k; i++){
		array[i] = d[i].q[0];
	}
	int cnt = 0;
	for(int i = 0; i < k; i++){
		if(a[array[i]].c[10] == 1){
			cnt++;
		}
	}
	if(cnt * 2 >= k){
		return 1;
	}
	return 0;
} 

void datafile_to_array(customer a[], int n, FILE *f){
	if(f == NULL){
		printf("Khong the mo file !\n"); 
	} 
	else{
		int i = 0, j = 0; 
		double p; 
		while(fscanf(f, "%lf", &p) != -1){
			if(i % 11 == 0){
				a[j].c[0] = p; 
			} 
			else if(i % 11 == 1){
				a[j].c[1] = p; 
			}
			else if(i % 11 == 2){
				a[j].c[2] = p; 
			}
			else if(i % 11 == 3){
				a[j].c[3] = p; 
			}
			else if(i % 11 == 4){
				a[j].c[4] = p; 
			}
			else if(i % 11 == 5){
				a[j].c[5] = p; 
			}
			else if(i % 11 == 6){
				a[j].c[6] = p; 
			}
			else if(i % 11 == 7){
				a[j].c[7] = p; 
			}
			else if(i % 11 == 8){
				a[j].c[8] = p; 
			}
			else if(i % 11 == 9){
				a[j].c[9] = p; 
			}
			else if(i % 11 == 10){
				a[j].c[10] = p; 
				j++;
			} 
			i++;  
		} 
	} 
}

void print(customer a[], int n){
	for(int k = 0; k < n; k++){
		printf("%6.0lf %3.0lf %2.0lf ", a[k].c[0], a[k].c[1], a[k].c[2]); 
		printf("%3.0lf %2.0lf %2.0lf ", a[k].c[3], a[k].c[4], a[k].c[5]);
		printf("%2.0lf %2.0lf ", a[k].c[6], a[k].c[7]);
		printf("%7.2lf %8.2lf ", a[k].c[8], a[k].c[9]);
		printf("%2.0lf\n", a[k].c[10]); 
	} 
} 

double Precision(customer a[], int p[], int n){
	int TP = 0;
	int Nh_p = 0;
	for(int i = 0; i < n; i++){
		if(p[i] == 1 && a[i].c[10] == 1){
			TP++;
			Nh_p++;  
		} 
		else if(p[i] == 1 && a[i].c[10] == 0){
			Nh_p++;  
		} 
	} 
	double precision = 1.0 * TP / Nh_p;
	return precision; 
} 

double Recall(customer a[], int p[], int n){
	int TP = 0;
	int N_p = 0;
	for(int i = 0; i < n; i++){
		if(p[i] == 1 && a[i].c[10] == 1){
			TP++; 
			N_p++; 
		}  
		else if(p[i] == 0 && a[i].c[10] == 1){
			N_p++; 
		}
	} 
	double recall = 1.0 * TP / N_p;
	return recall; 
} 

double Accuracy(customer a[], int p[], int n){
	int TP = 0, TN = 0, FP = 0, FN = 0;   
	for(int i = 0; i < n; i++){
		if(p[i] == 1 && a[i].c[10] == 1){
			TP++;
		} 
		else if(p[i] == 1 && a[i].c[10] == 0){
			FP++; 
		} 
		else if(p[i] == 0 && a[i].c[10] == 1){
			FN++; 
		}
		else if(p[i] == 0 && a[i].c[10] == 0){
			TN++;  
		}
	} 
	double accuracy = (double)(TN + TP) / (TN + TP + FN + FP);
	return accuracy; 
} 

double F1_score(customer a[], int p[], int n){
	double P = Precision(a, p, n);
	double R = Recall(a, p, n);
	double f1_score = 2 * P * R / (P + R);
	return f1_score; 
} 

void Confusion_matrix(customer a[], int p[], int n){
	int TP = 0, TN = 0, FP = 0, FN = 0; 
	int Nh_p = 0, Nh_m = 0, N_p = 0, N_m = 0;  
	for(int i = 0; i < n; i++){
		if(p[i] == 1 && a[i].c[10] == 1){
			TP++;
			Nh_p++; 
			N_p++; 
		} 
		else if(p[i] == 1 && a[i].c[10] == 0){
			FP++; 
			Nh_p++; 
			N_m++; 
		} 
		else if(p[i] == 0 && a[i].c[10] == 1){
			FN++; 
			Nh_m++; 
			N_p++; 
		}
		else if(p[i] == 0 && a[i].c[10] == 0){
			TN++; 
			Nh_m++;
			N_m++; 
		}
	} 
	printf("Confusion matrix :\n"); 
	printf(" ___________________________________________________\n");
	printf("|                  |        Truth        |          |\n");
	printf("|                  |_____________________|          |\n");
	printf("|                  |    1     |    0     |          |\n");
	printf("|------------------|----------|----------|----------|\n");
	printf("|            |  1  |   %3d    |   %3d    |   %3d    |\n", TP, FP, Nh_p);
	printf("|  Estimate  |-----|----------|----------|----------|\n");
	printf("|            |  0  |   %3d    |   %3d    |   %3d    |\n", FN, TN, Nh_m);
	printf("|------------------|----------|----------|----------|\n");
	printf("|                  |   %3d    |   %3d    |  %4d    |\n", N_p, N_m, n);
	printf("|__________________|__________|__________|__________|\n");
}

int main(){
	customer dataset[9000];
	customer datatest[1000];
	FILE *f;
	f = fopen("customer_dataset.txt", "r");
	datafile_to_array(dataset, 9000, f); 
	fclose(f);
	FILE *p; 
	p = fopen("customer_datatest.txt", "r");
	datafile_to_array(datatest, 1000, p);
	fclose(p);
	feature_scaling(dataset, 9000);
	feature_scaling(datatest, 1000);
	double precision[50]; 
	double recall[50]; 
	double f1_score[50]; 
	double accuracy[50];
	for(int k = 0; k < 50; k++){
		int *pre = (int*)malloc(1000 * sizeof(int));
		for(int i = 0; i < 1000; i++){
			pre[i] = predict(dataset, 9000, datatest[i], k + 1);
		}
		precision[k] = Precision(datatest, pre, 1000);
		recall[k] = Recall(datatest, pre, 1000);
		f1_score[k] = F1_score(datatest, pre, 1000);
		accuracy[k] = Accuracy(datatest, pre, 1000);
		free(pre); 
	} 
	FILE *x;
	x = fopen("precision.txt", "w");
	for(int i = 0; i < 50; i++){
		fprintf(x, "%d %6.4lf\n", i, precision[i]);
	} 
	fclose(x);
	FILE *y;
	y = fopen("recall.txt", "w");
	for(int i = 0; i < 50; i++){
		fprintf(y, "%d %6.4lf\n", i, recall[i]);
	} 
	fclose(y);
	FILE *z;
	z = fopen("f1_score.txt", "w");
	for(int i = 0; i < 50; i++){
		fprintf(z, "%d %6.4lf\n", i, f1_score[i]);
	} 
	fclose(z);
	FILE *t;
	t = fopen("accuracy.txt", "w");
	for(int i = 0; i < 50; i++){
		fprintf(t, "%d %6.4lf\n", i, accuracy[i]);
	} 
	fclose(t);
	int pre[1000];
	for(int i = 0; i < 1000; i++){
		pre[i] = predict(dataset, 9000, datatest[i], 40);
	}
	printf("Voi K = 40\n");
	printf("Precision : %.4lf\n", Precision(datatest, pre, 1000)); 
	printf("Recall : %.4lf\n", Recall(datatest, pre, 1000));
	printf("F1-score : %.4lf\n", F1_score(datatest, pre, 1000)); 
	printf("Accuracy : %.4lf\n", Accuracy(datatest, pre, 1000)); 
	Confusion_matrix(datatest, pre, 1000);
	return 0; 
}
