#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>

#define float double
using namespace std;

struct Layer{
    string activation_function;
    int neuron;
};
class Optimizer{
    public:
    // here adam optimizer only implemented
    vector<vector<vector<float>>> m_w,v_w;
    vector<vector<float>> m_b,v_b;
    float b1 = 0.9,b2 = 0.99,eps = 1e-8,lr = 0.01;
    Optimizer(vector<Layer>& layers){
        int n = layers.size();

        m_w.resize(n-1);v_w.resize(n-1);
        m_b.resize(n-1);v_b.resize(n-1);

        for(int i=0;i+1<n;++i){
            int r = layers[i].neuron,c = layers[i+1].neuron;
            m_w[i] = v_w[i] = vector<vector<float>> (r,vector<float>(c));
            m_b[i] = v_b[i] = vector<float> (c);
        }
    }
    float adam(int i,int j,int k,float w,float dw){
        m_w[i][j][k] = b1 * m_w[i][j][k] + (1 - b1) * dw;
        v_w[i][j][k] = b2 * v_w[i][j][k] + (1 - b2) * dw * dw;

        w -= lr * (m_w[i][j][k] / (sqrt(v_w[i][j][k]) + eps));
        return w;
    }
    float adam(int i,int j,float b,float db){
        m_b[i][j] = b1 * m_b[i][j] + (1 - b1) * db;
        v_b[i][j] = b2 * v_b[i][j] + (1 - b2) * db * db;

        b -= lr * (m_b[i][j] / (sqrt(v_b[i][j]) + eps));
        return b;
    }
};
class Activations{
    public:
    // all the activation functions are implemented
    float activation(float z,string s){
        if(s == "linear") return z;
        if(s == "relu") return max(0.0,z);
        if(s == "tanh") return tanh(z);
        if(s == "sigmoid") return 1.0 / (1.0 + exp(-z));
        return 0.0;
    }
    // derivaties of activation function
    float d_activation(float z,string s){
        if (s == "linear") return 1.0;
        if (s == "relu") return (z > 0.0) ? 1.0 : 0.0;
        if (s == "tanh") {
            float tanh_z = tanh(z);
            return 1.0 - tanh_z * tanh_z;
        }
        if (s == "sigmoid") {
            float sigmoid_z = 1.0 / (1.0 + exp(-z));
            return sigmoid_z * (1.0 - sigmoid_z);
        }
        return 0.0;
    }
};
void initialize(vector<vector<vector<float>>>& w,vector<vector<float>>& b,vector<Layer>& layers){
    // Initialization of the weight and bias for each layer
    int n = w.size(),n_in = layers[0].neuron,n_out = layers[n].neuron;
    float stddev;
    srand(time(0));
    for(int i=0;i<n;++i){
        if(layers[i+1].activation_function == "linear"){
            stddev = 1;
        }
        else if(layers[i+1].activation_function == "relu"){
            stddev = sqrt(2.0 / n_in);
        }
        else{
            stddev = sqrt((2.0 / (n_in + n_out)));
        }
        int r = layers[i].neuron,c = layers[i+1].neuron;
        w[i] = vector<vector<float>> (r,vector<float>(c));
        b[i] = vector<float> (c);
        for(int j=0;j<r;++j){
            for(int k=0;k<c;++k){
                w[i][j][k] = stddev * (static_cast<float>(rand()) / RAND_MAX);
            }
        }
        for(int k=0;k<c;++k){
            b[i][k] = stddev * (static_cast<float>(rand()) / RAND_MAX);
        }
    }
}
float d_cost_function(float y,float pred,string type){
    // derivation of the cost function or loss function
    float cost = 0;
    if(type == "sigmoid"){
        pred = max(pred,1e-15);
        cost = ((1 - y) / (1 - pred)) - (y / pred);
    }
    else{
        cost = 2 * (pred - y);
    }
    return cost;
}
float loss_function(float y,float pred,string type){
    // loss function used to calculate the over all loss
    float loss = 0;
    if(type == "sigmoid"){
        loss = -((y*log(pred)) + ((1-y)*log(1-pred)));
    }
    else{
        loss = (pred - y) * (pred - y);
    }
    return loss;
}
float func(vector<float>& v){
    //create a function for testcase generator
    float x = v[0];
    return (x > 500 ? 1 : 0);
    //return (x * 2);
}
int main(){
    int n,N,epochs,in_dim;
    n = 4;//No of layers
    N = 1000;// No of input test
    epochs = 1000;// No of epochs
    in_dim = 1;//No of feature or input in first layer
    //vector<Layer> layers = {{"linear",in_dim}, {"relu",4}, {"relu",3}, {"linear",1}};
    vector<Layer> layers = {{"linear",in_dim},{"sigmoid",4},{"sigmoid",3},{"sigmoid",1}};

    vector<vector<float>> input(N,vector<float>(in_dim));
    vector<float> output(N);

    srand(time(0));
    for(int i=0;i<N;++i){
        for(int j=0;j<in_dim;++j){
            input[i][j] = rand() % 1000;
        }
        output[i] = func(input[i]);
    }
    vector<vector<vector<float>>> w(n-1);
    vector<vector<float>> b(n-1);

    initialize(w,b,layers);
    Activations at;
    Optimizer opt(layers);
    
    for(int epoch = 0;epoch < epochs; ++epoch){
        float total_loss = 0;

        vector<vector<vector<float>>> deltaw(n-1);
        vector<vector<float>> deltab(n-1);
        
        for(int i=0;i+1<n;++i){
            int r = layers[i].neuron,c = layers[i+1].neuron;
            deltaw[i] = vector<vector<float>> (r,vector<float>(c));
            deltab[i] = vector<float> (c);
        }

        for(int i=0;i<N;++i){
            vector<vector<float>> z(n),a(n);
            for(int i=0;i<n;++i){
                z[i] = a[i] = vector<float>(layers[i].neuron);
            }
            for(int j=0;j<in_dim;++j){
                a[0][j] = input[i][j];
            }
            float y = output[i]; // actual value
            //forward pass
            for(int i=0;i+1<n;++i){
                int r = w[i].size(),c = w[i][0].size();
                // r -> prev layer c -> next layer
                for(int j=0;j<r;++j){
                    for(int k=0;k<c;++k){
                        z[i+1][k] += a[i][j] * w[i][j][k];
                    }
                }
                for(int k=0;k<c;++k){
                    a[i+1][k] = at.activation(z[i+1][k] + b[i][k],layers[i+1].activation_function);
                }
                // here b[i][k] why i+1 means b size n-1 so it already points the next layer
            }

            //backward pass

            float cost = d_cost_function(y,a[n-1][0],layers[n-1].activation_function);
            vector<float> prev(1,1);// always one neuron in the output layer its not a multiclass or multilabel classification
            prev[0] = at.d_activation(z[n-1][0],layers[n-1].activation_function);

            for(int i=n-2;i>=0;--i){
                int r = w[i].size(),c = w[i][0].size();
                // r -> next layer c -> prev layer
                vector<float> next(r);
                for(int j=0;j<r;++j){
                    for(int k=0;k<c;++k){
                        next[j] += prev[c] * w[i][j][k] * at.d_activation(z[i][j],layers[i].activation_function);
                    }
                }

                for(int j=0;j<r;++j){
                    for(int k=0;k<c;++k){
                        deltaw[i][j][k] += cost * prev[k] * z[i][k];
                    }
                }
                for(int k=0;k<c;++k){
                    deltab[i][k] += cost * prev[k];
                }
            }
            total_loss += loss_function(y,a[n-1][0],layers[n-1].activation_function);
        }
        for(int i=0;i+1<n;++i){
            int r = w[i].size(),c = w[i][0].size();
            for(int j=0;j<r;++j){
                for(int k=0;k<c;++k){
                    float dw = deltaw[i][j][k] / N;
                    w[i][j][k] = opt.adam(i,j,k,w[i][j][k],dw);
                }
            }
            for(int k=0;k<c;++k){
                float db = deltab[i][k] / N;
                b[i][k] = opt.adam(i,k,b[i][k],db);
            }
        }
        if(epoch%100 == 0){
            cout << "Epoch[" << epoch << "/" << epochs << "]" << " Total Loss: " << total_loss/N << endl;
        }
    }
    //check
    vector<float> x = {1200};
    vector<vector<float>> z(n),a(n);
    for(int i=0;i<n;++i){
        z[i] = a[i] = vector<float>(layers[i].neuron);
    }
    for(int j=0;j<in_dim;++j){
        a[0][j] = x[j];
    }
    for(int i=0;i+1<n;++i){
        int r = w[i].size(),c = w[i][0].size();
        // r -> prev layer c -> next layer
        for(int j=0;j<r;++j){
            for(int k=0;k<c;++k){
                z[i+1][k] += a[i][j] * w[i][j][k];
            }
        }
        for(int k=0;k<c;++k){
            a[i+1][k] = at.activation(z[i+1][k] + b[i][k],layers[i+1].activation_function);
        }
        // here b[i][k] why i+1 means b size n-1 so it already points the next layer
    }
    cout << "Actual value for Input " << x[0] << ": " << func(x) << endl;
    cout << "Prediction for Input " << x[0] << ": " << a[n-1][0] << endl;
}