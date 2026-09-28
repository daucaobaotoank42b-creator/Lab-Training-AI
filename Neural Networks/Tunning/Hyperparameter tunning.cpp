#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h> 

double **matrix(int m, int n){
    double **a = (double**)malloc(m * sizeof(double*));
    for(int i = 0; i < m; i++){
        a[i] = (double*)malloc(n * sizeof(double));
    }
    return a;
}

void free_matrix(double **a, int m){
    for(int i = 0; i < m; i++){
    	free(a[i]);
	} 
    free(a);
}

double **zero_matrix(int m, int n){
    double **Z = matrix(m, n);
    for(int i = 0; i < m; i++){
    	for(int j = 0; j < n; j++){
    		Z[i][j] = 0;
		}  
	} 
    return Z;
}

void equal(double **a, int m, int n, double **b){
    for(int i = 0; i < m; i++){
    	for(int j = 0; j < n; j++){
    		a[i][j] = b[i][j];
		}
	}      
}

double **matrix_multi(double **a, int m, int n, double **b, int q){
    double **c = zero_matrix(m, q);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < q; j++){
            for(int k = 0; k < n; k++){
                c[i][j] += a[i][k] * b[k][j];  
            }                
        }
    }
    return c;
}

double **scalar_multi(double **a, int m, int n, double x){
    double **X = matrix(m, n);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            X[i][j] = x * a[i][j];
        }           
    }
    return X;
}


double **transpose(double **a, int m, int n){
    double **aT = matrix(n, m);
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            aT[i][j] = a[j][i]; 
        }          
    }
    return aT;
}

double **add_bias(double **a, int m, int n, double **b){
    double **c = matrix(m, n);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            c[i][j] = a[i][j] + b[i][0];
        }
    }       
    return c;
}

double **matrix_add(double **a, int m, int n, double **b){
    double **c = matrix(m, n);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            c[i][j] = a[i][j] + b[i][j];
        }
    }      
    return c;
}

double **matrix_minus(double **a, int m, int n, double **b){
    double **c = matrix(m, n);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            c[i][j] = a[i][j] - b[i][j];
        }
    }     
    return c;
}

double min_arr(double *x, int n){
    double mi = x[0];
    for(int i = 1; i < n; i++){
        if(x[i] < mi){
            mi = x[i];
        } 
    } 
    return mi;
}

double max_arr(double *x, int n){
    double ma = x[0];
    for(int i = 1; i < n; i++){
       if(x[i] > ma){
           ma = x[i];
       }  
    } 
    return ma;
}

void scale(double **x, int m, int n, double **mi_ma){
	for(int i = 0; i < m; i++){
		mi_ma[i][0] = min_arr(x[i], n);
		mi_ma[i][1] = max_arr(x[i], n); 
	} 
    for(int i = 0; i < m; i++){
    	for(int j = 0; j < n; j++){
    		x[i][j] = (x[i][j] - mi_ma[i][0]) / (mi_ma[i][1] - mi_ma[i][0]);
		} 
    } 
}

void restore_1d(double *x, int n, double mi, double ma){
    for(int i = 0; i < n; i++){
        x[i] = x[i] * (ma - mi) + mi;
    } 
}

void apply_scale(double **x, int m, int n, double **mi_ma){ 
    for(int i = 0; i < m; i++){
    	for(int j = 0; j < n; j++){
    		x[i][j] = (x[i][j] - mi_ma[i][0]) / (mi_ma[i][1] - mi_ma[i][0]);
		} 
    } 
}

double **one_hot(double *a, int output_dim, int n_train){
	double **onehot = zero_matrix(output_dim, n_train);
	for(int i = 0; i < n_train; i++){
		for(int j = 0; j < output_dim; j++){
			if(j == a[i]){
				onehot[j][i] = 1; 
			} 
		} 
	} 
	return onehot; 
} 

double **onehot(double **a, int m, int n){ 	
	double **transA = transpose(a, m, n); 
	double *b = (double*)malloc(n * sizeof(double)); 
	for(int i = 0; i < n; i++){
		b[i] = max_arr(transA[i], m); 
	} 
	free_matrix(transA, n); 
	for(int i = 0; i < n; i++){
		int src = 1; 
		for(int j = 0; j < m; j++){
			if(a[j][i] == b[i] && src == 1){
				a[j][i] = 1; 
				src = 0; 
			}  
			else{
				a[j][i] = 0;
			} 
		} 
	} 
	free(b); 
	return a; 
} 

double *restore_onehot(double **a, int m, int n){
	double **tmp = onehot(a, m, n);
	double *c = (double*)malloc(n * sizeof(double)); 
	for(int i = 0; i < n; i++){
		for(int j = 0; j < m; j++){
			if(tmp[j][i] == 1){
				c[i] = j; 
			} 
		} 
	} 
	return c; 
} 

double LeakyReLU(double x){
    return (x > 0) ? x : 0.1 * x;
}

double gradLeakyReLU(double x){
    return (x > 0) ? 1.0 : 0.1;   
}

double **activation_matrix_LeakyReLU(double **a, int m, int n){
    double **b = matrix(m, n);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            b[i][j] = LeakyReLU(a[i][j]);
        }
    }     
    return b;
}

double Sigmoid(double x){
	return 1.0 / (1 + exp(-x)); 
} 

double gradSigmoid(double x){
	return Sigmoid(x) * (1 - Sigmoid(x)); 
} 

double **activation_matrix_Sigmoid(double **a, int m, int n){
    double **b = matrix(m, n);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            b[i][j] = Sigmoid(a[i][j]);
        }
    }    
    return b;
} 

double **activation_matrix_Softmax(double **a, int m, int n){
	double **b = matrix(m, n); 
	for(int j = 0; j < n; j++){
		double sum = 0; 
		for(int i = 0; i < m; i++){
			sum += exp(a[i][j]); 
		}
		for(int i = 0; i < m; i++){
			b[i][j] = exp(a[i][j]) / sum; 
		} 
	} 
	return b; 
} 

struct Layer{
    int size_in;
    int size_out;
    double **w;   // (size_in  x size_out) 
    double **b;   // (size_out x 1)       
    double **z;   // (size_out x n_train)  
    double **a;   // (size_out x n_train)  
    double **e;   // (size_out x n_train)  
    double **dw;  // (size_in  x size_out) 
    double **db;  // (size_out x 1) 
    double **mw, **vw;   // (size_in  x size_out) 
    double **mb, **vb;   // (size_out x 1) 
};

typedef struct Layer Layer;

struct NeuralNetworks{
    int layer;        // so luong layer 
    int n_train;      // so mau du lieu train 
    int *adam_t;
    Layer *Layers;
};

typedef struct NeuralNetworks NeuralNetworks;

void random_matrix(double **x, int m, int n, double limit){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            x[i][j] = (2.0 * rand() / RAND_MAX - 1) * limit;
        }
    } 
}

NeuralNetworks create_network(int input_dim, int *hidden_sizes, int num_hidden, int output_dim, int n_train){
    NeuralNetworks net;
    net.layer = num_hidden + 1;  
    net.n_train = n_train;
    net.adam_t = (int*)malloc(sizeof(int));
    *net.adam_t = 0;
    net.Layers = (Layer*)malloc(net.layer * sizeof(Layer));
    int prev_size = input_dim;
    for(int i = 0; i < net.layer; i++){
        int this_size = (i < num_hidden) ? hidden_sizes[i] : output_dim;
        Layer *L = &net.Layers[i];
        L->size_in  = prev_size;
        L->size_out = this_size;
        L->w  = matrix(L->size_in, L->size_out);
        L->b  = matrix(L->size_out, 1);
        L->z  = matrix(L->size_out, n_train);
        L->a  = matrix(L->size_out, n_train);
        L->e  = matrix(L->size_out, n_train);
        L->dw = matrix(L->size_in, L->size_out);
        L->db = matrix(L->size_out, 1);
        L->mw = zero_matrix(L->size_in, L->size_out);
        L->vw = zero_matrix(L->size_in, L->size_out);
        L->mb = zero_matrix(L->size_out, 1);
        L->vb = zero_matrix(L->size_out, 1);
        double limit = sqrt(2.0 / L->size_in);
        random_matrix(L->w, L->size_in, L->size_out, limit);
        for(int k = 0; k < L->size_out; k++){
        	L->b[k][0] = 0;
		} 
        prev_size = this_size;
    }
    return net;
}

void free_network(NeuralNetworks net){
    for(int i = 0; i < net.layer; i++){
        Layer L = net.Layers[i];
        free_matrix(L.w, L.size_in);
        free_matrix(L.b, L.size_out);
        free_matrix(L.z, L.size_out);
        free_matrix(L.a, L.size_out);
        free_matrix(L.e, L.size_out);
        free_matrix(L.dw, L.size_in);
        free_matrix(L.db, L.size_out);
        free_matrix(L.mw, L.size_in);
        free_matrix(L.vw, L.size_in);
        free_matrix(L.mb, L.size_out);
        free_matrix(L.vb, L.size_out);
    }
    free(net.adam_t);
    free(net.Layers);
}

void feedforward_classification_train(NeuralNetworks net, double **input, int n){
    double **prev_a = input;
    for(int i = 0; i < net.layer; i++){
        Layer *L = &net.Layers[i];
        double **wT = transpose(L->w, L->size_in, L->size_out);
        double **z1 = matrix_multi(wT, L->size_out, L->size_in, prev_a, n);
        double **z2 = add_bias(z1, L->size_out, n, L->b);
        equal(L->z, L->size_out, n, z2);
        free_matrix(wT, L->size_out);
        free_matrix(z1, L->size_out);
        free_matrix(z2, L->size_out);
        if(i == net.layer - 1){
        	double **z3 = activation_matrix_Softmax(L->z, L->size_out, n); 
            equal(L->a, L->size_out, n, z3);
            free_matrix(z3, L->size_out); 
        } 
		else {
            double **act = activation_matrix_LeakyReLU(L->z, L->size_out, n);
            equal(L->a, L->size_out, n, act);
            free_matrix(act, L->size_out);
        }
        prev_a = L->a;
    }
}

void backpropagation_classification(NeuralNetworks net, double **input, double **y, int n){
	for(int i = net.layer - 1; i >= 0; i--){
		Layer *L = &net.Layers[i];
        if(i == net.layer - 1){
            for(int k = 0; k < L->size_out; k++){
            	for(int q = 0; q < n; q++){
            		L->e[k][q] = (L->a[k][q] - y[k][q]) / n;
				}
			}                   
        } 
		else {
            Layer *Lnext = &net.Layers[i + 1];
            for(int j = 0; j < L->size_out; j++){
                for(int q = 0; q < n; q++){
                    double sum = 0;
                    for(int k = 0; k < Lnext->size_out; k++){
                    	sum += Lnext->e[k][q] * Lnext->w[j][k];
					}  
                    L->e[j][q] = gradLeakyReLU(L->z[j][q]) * sum;
                }
            }
        }
        double **prev_a = (i == 0) ? input : net.Layers[i - 1].a;
        double **eT = transpose(L->e, L->size_out, n);
        double **t  = matrix_multi(prev_a, L->size_in, n, eT, L->size_out);
        equal(L->dw, L->size_in, L->size_out, t);
        free_matrix(eT, n);
        free_matrix(t, L->size_in);
        for(int k = 0; k < L->size_out; k++){
            double s = 0;
            for(int q = 0; q < n; q++){
            	s += L->e[k][q];
			} 
            L->db[k][0] = s;
		}
	} 
} 

#define ADAM_BETA1 0.9001 
#define ADAM_BETA2 0.999
#define ADAM_EPS   1e-8 

void adam_step(double **theta, double **grad, double **m, double **v, int rows, int cols, double alpha, int t){
    double beta1_t = pow(ADAM_BETA1, t);
    double beta2_t = pow(ADAM_BETA2, t);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            double g = grad[i][j];
            m[i][j] = ADAM_BETA1 * m[i][j] + (1 - ADAM_BETA1) * g;
            v[i][j] = ADAM_BETA2 * v[i][j] + (1 - ADAM_BETA2) * g * g;
            double m_hat = m[i][j] / (1 - beta1_t);
            double v_hat = v[i][j] / (1 - beta2_t);
            theta[i][j] -= alpha * m_hat / (sqrt(v_hat) + ADAM_EPS);
        }
    }
}

void update(NeuralNetworks net, double alpha){
    *net.adam_t += 1;
    for(int i = 0; i < net.layer; i++){
        Layer *L = &net.Layers[i];
        adam_step(L->w, L->dw, L->mw, L->vw, L->size_in, L->size_out, alpha, *net.adam_t);
        adam_step(L->b, L->db, L->mb, L->vb, L->size_out, 1, alpha, *net.adam_t);
    }
}

double cost_cross_entropy(double **y, double **yhat, int m, int n){
	double sum = 0;
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			sum += y[i][j] * log(yhat[i][j] + 1e-10); 
		} 
	} 
	return -sum / n; 
} 

void shuffle_index(int *idx, int n){
    for(int i = n - 1; i > 0; i--){
        int j = rand() % (i + 1);
        int tmp = idx[i];
        idx[i] = idx[j];
        idx[j] = tmp;
    }
}

void shuffle_data(double **input, double **target, int input_dim, int output_dim, int n, int *index){
    double **new_input  = matrix(input_dim, n);
    double **new_target = matrix(output_dim, n);
    for(int i = 0; i < n; i++){
        int src = index[i];
        for(int j = 0; j < input_dim; j++){
        	new_input[j][i]  = input[j][src];
		}  
        for(int j = 0; j < output_dim; j++){
        	new_target[j][i] = target[j][src];
		} 
    }
    equal(input,  input_dim,  n, new_input);
    equal(target, output_dim, n, new_target);
    free_matrix(new_input, input_dim);
    free_matrix(new_target, output_dim);
}

double **predict_classification(NeuralNetworks, double **, int);

void train_classification(NeuralNetworks net, double **input, double **y, int in_dim, int out_dim, 
double alpha, int max_epoch, double target_cost, int batch_size, double **output_truth_test, double **input_test, int n_test){
	int n = net.n_train; 
	int *index = (int*)malloc(n * sizeof(int));
	for(int i = 0; i < n; i++){
		index[i] = i; 
	} 
	FILE *f;
	f = fopen("Training_cost.txt", "w"); 
	FILE *p;
	p = fopen("Validation_cost.txt", "w"); 
	for(int epoch = 0; epoch < max_epoch; epoch++){
		shuffle_index(index, n);
		shuffle_data(input, y, in_dim, out_dim, n, index);		
		for(int start = 0; start < n; start += batch_size){
			int b = batch_size < (n - start) ? batch_size : (n - start);
			double **batch_in = matrix(in_dim, b); 
			double **batch_out = matrix(out_dim, b);
			for(int j = 0; j < b; j++){
				for(int q = 0; q < in_dim; q++){
					batch_in[q][j] = input[q][j + start];  
				} 
				for(int q = 0; q < out_dim; q++){
					batch_out[q][j] = y[q][j + start]; 
				} 
			} 
			feedforward_classification_train(net, batch_in, b);
			backpropagation_classification(net, batch_in, batch_out, b);
			update(net, alpha);
			free_matrix(batch_in, in_dim);
			free_matrix(batch_out, out_dim); 
		} 
		feedforward_classification_train(net, input, n);
		double cost = cost_cross_entropy(y, net.Layers[net.layer - 1].a, out_dim, net.n_train);	
		if(isnan(cost) || isinf(cost)){
            printf("[Dung huan luyen] Phat hien phan ky (NaN/Inf) tai epoch %d.\n", epoch);
            return;
        } 			
        if(epoch % 100== 0){
        	printf("%5d   %10.8lf\n", epoch, cost);
        	fprintf(f, "%5d   %10.8lf\n", epoch / 100, 100 * cost); 
        	fprintf(p, "%5d   %10.8lf\n", epoch / 100, 100 * cost_cross_entropy(output_truth_test, predict_classification(net, input_test, n_test), out_dim, n_test));
		}
        if(cost < target_cost){
            printf("Hoi tu tai epoch %d | cost = %.8lf\n", epoch, cost);
            printf("Cost validation = %.8lf\n", cost_cross_entropy(output_truth_test, predict_classification(net, input_test, n_test), out_dim, n_test));
            return;
        } 
	} 
	fclose(f); 
	fclose(p); 
	printf("Dat toi max_epoch, cost train cuoi = %.8lf\n", cost_cross_entropy(y, net.Layers[net.layer - 1].a, out_dim, net.n_train));
	printf("Dat toi max_epoch, cost test cuoi = %.8lf\n", cost_cross_entropy(output_truth_test, predict_classification(net, input_test, n_test), out_dim, n_test));
} 

double **predict_classification(NeuralNetworks net, double **input, int n_points){
	double **prev_a = input;
    int owns_prev = 0; 
    for(int i = 0; i < net.layer; i++){
        Layer *L = &net.Layers[i];
        double **wT = transpose(L->w, L->size_in, L->size_out);
        double **z1 = matrix_multi(wT, L->size_out, L->size_in, prev_a, n_points);
        double **z2 = add_bias(z1, L->size_out, n_points, L->b);
        free_matrix(wT, L->size_out);
        free_matrix(z1, L->size_out);
        double **a_layer;
        if(i == net.layer - 1){
            a_layer = activation_matrix_Softmax(z2, L->size_out, n_points);  
            free_matrix(z2, L->size_out);
        } 
		else{
            a_layer = activation_matrix_LeakyReLU(z2, L->size_out, n_points);
            free_matrix(z2, L->size_out);
        }
        if(owns_prev){
        	free_matrix(prev_a, L->size_in);
		} 
        prev_a = a_layer;
        owns_prev = 1;
    }
    return prev_a;
} 

double Precision(double **truth, double **pre, int dim, int size){
    double precision = 0;
    int *weight = (int*)malloc(dim * sizeof(int));
	for(int i = 0; i < dim; i++){
		weight[i] = 0; 
	} 
    for(int i = 0; i < dim; i++){
		int TP = 0;
		int Nh_p = 0;
        for(int j = 0; j < size; j++){
        	if(truth[i][j] == 1){
        		weight[i]++;
			} 
		    if(truth[i][j] == 1 && pre[i][j] == 1){
			    TP++;
		    	Nh_p++;  
		    } 
	    	else if(truth[i][j] == 0 && pre[i][j] == 1){
	    		Nh_p++;  
	    	}             
        }
        precision += 1.0 * weight[i] * TP / (size * Nh_p + 1e-10);
    }	
    free(weight); 
	return precision; 
} 

double Recall(double **truth, double **pre, int dim, int size){
    double recall = 0;
    int *weight = (int*)malloc(dim * sizeof(int));
	for(int i = 0; i < dim; i++){
		weight[i] = 0; 
	} 
    for(int i = 0; i < dim; i++){
		int TP = 0;
		int N_p = 0;
        for(int j = 0; j < size; j++){
        	if(truth[i][j] == 1){
        		weight[i]++;
			} 
		    if(truth[i][j] == 1 && pre[i][j] == 1){
			    TP++;
		    	N_p++;  
		    } 
	    	else if(truth[i][j] == 1 && pre[i][j] == 0){
	    		N_p++;  
	    	}             
        }
        recall += 1.0 * weight[i] * TP / (size * N_p + 1e-10);
    }	
    free(weight); 
	return recall; 
} 

double Accuracy(double **truth, double **pre, int dim, int size){
 	int x = 0;   
    for(int j = 0; j < size; j++){
 		for(int i = 0; i < dim; i++){
 			if(truth[i][j] == 1 && pre[i][j] == 1){
 				x++;
 			}  			
		}
 	} 
 	double accuracy = 1.0 * x / size;
 	return accuracy; 
} 

double F1_score(double **truth, double **pre, int dim, int size){
 	double P = Precision(truth, pre, dim, size);
 	double R = Recall(truth, pre, dim, size);
 	double f1_score = 2 * P * R / (P + R + 1e-10);
 	return f1_score; 
} 

void Confusion_matrix(double **truth, double **pre, int dim, int size){
	double **truthT = transpose(truth, dim, size); 
 	double **confusion = matrix_multi(pre, dim, size, truthT, dim); 
 	printf("Confusion matrix :\n");
 	for(int i = 0; i < dim; i++){
 		for(int j = 0; j < dim; j++){
 			printf("|  %3d  |", (int)confusion[i][j]); 
		} 
		printf("\n"); 
	} 
	free_matrix(truthT, size);
	free_matrix(confusion, dim); 
}

int main(){
    srand((unsigned int)time(NULL));
    int n_train = 3000;
    int input_dim = 11;
	int output_dim = 4; 
    double **input_train = matrix(input_dim, n_train);
    double *label_train_raw = (double*)malloc(n_train * sizeof(double)); 
    FILE *f = fopen("White_wine_train.txt", "r");
    if(f == NULL){
        printf("Khong tim thay file du lieu\n");
        return 0; 
    } 
	else{
        for(int i = 0; i < n_train; i++){
        	for(int j = 0; j < input_dim; j++){
        		fscanf(f, "%lf", &input_train[j][i]); 
			} 
            fscanf(f, "%lf", &label_train_raw[i]);
        }
        fclose(f);
    }
    double **min_max = matrix(input_dim, 2); 
    scale(input_train, input_dim, n_train, min_max); 
    double **target = matrix(output_dim, n_train); 
    double **tmp = one_hot(label_train_raw, output_dim, n_train);
    equal(target, output_dim, n_train, tmp);  
    int hidden_sizes[] = {8, 8};
    NeuralNetworks net = create_network(input_dim, hidden_sizes, 2, output_dim, n_train);
    printf("Bat dau huan luyen ...\n"); 
    int n_test = 961;
    double **input_test = matrix(input_dim, n_test); 
    double *label_test_raw = (double*)malloc(n_test * sizeof(double));
	FILE *p;
	p = fopen("White_wine_test.txt", "r");
    if(p == NULL){
        printf("Khong tim thay file du lieu\n");
        return 0; 
    } 
	else{
        for(int i = 0; i < n_test; i++){
        	for(int j = 0; j < input_dim; j++){
        		fscanf(p, "%lf", &input_test[j][i]); 
			} 
            fscanf(p, "%lf", &label_test_raw[i]);
        }
        fclose(p);
    }	 
    apply_scale(input_test, input_dim, n_test, min_max);    
    double **z_truth = one_hot(label_test_raw, output_dim, n_test); 
    train_classification(net, input_train, target, input_dim, output_dim, 0.006, 50000, 0.02, 128, z_truth, input_test, n_test);
    
    double **z_pred = predict_classification(net, input_test, n_test);
    double **z_onehot = onehot(z_pred, output_dim, n_test); 
    double *z_out = restore_onehot(z_onehot, output_dim, n_test); 
      
	printf("Precision : %lf\n", Precision(z_truth, z_onehot, output_dim, n_test));
	printf("Recall : %lf\n", Recall(z_truth, z_onehot, output_dim, n_test));
	printf("Accuracy : %lf\n", Accuracy(z_truth, z_onehot, output_dim, n_test));
	printf("F1-score : %lf\n", F1_score(z_truth, z_onehot, output_dim, n_test));
	Confusion_matrix(z_truth, z_onehot, output_dim, n_test);
	free_network(net);
    free_matrix(input_train, input_dim);
    free_matrix(input_test, input_dim);
    free_matrix(target, output_dim); 
	free_matrix(tmp, output_dim); 
	free(label_train_raw); 
    free(label_test_raw);
    free_matrix(z_pred, output_dim);
	 
}
