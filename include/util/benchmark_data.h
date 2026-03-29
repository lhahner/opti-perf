#pragma once
#include <sstream>
#include <string>

/**
 * @brief Stores a single benchmark result row and provides CSV serialization helpers.
 */
class BenchmarkData
{
public:
    std::string timestamp = "";
    std::string device_name = "";
    std::string framework = "";
    std::string workload_name = "";
    std::string workload_type = "";
    std::string device = "";
    int batch_size = 0;
    long input_size = 0;
    std::string optimizer = "";
    float learning_rate = 0.0f;
    float beta1 = 0.0f;
    float beta2 = 0.0f;
    float epsilon = 0.0f;
    float time_ms = 0.0f;
    int batch_index = 0;
    float loss = 0.0f;
    float accuracy = 0.0f;
    std::string log_filename = "benchmarks-logs.csv";

    BenchmarkData()
    {
    }

    /**
     * @brief Constructs a populated benchmark data object.
     * @param ts Timestamp associated with the measurement.
     * @param device_name Human-readable device name.
     * @param fw Execution framework name.
     * @param w_name Workload name.
     * @param w_type Workload category or phase.
     * @param dev Device category.
     * @param b_size Batch size.
     * @param in_size Input size.
     * @param opt Optimizer name.
     * @param lr Learning rate.
     * @param b1 Beta1 coefficient.
     * @param b2 Beta2 coefficient.
     * @param eps Numerical stability epsilon.
     * @param t_ms Execution time in milliseconds.
     * @param b_index Batch index.
     * @param l Loss value.
     */
    BenchmarkData(const char *ts, const char *device_name, const char *fw, const char *w_name, const char *w_type, const char *dev,
                  int b_size, long in_size,
                  const char *opt, float lr, float b1, float b2,
                  float eps, float t_ms, int b_index, float l, float acc = 0.0f, const char *log_file = "benchmarks-logs.csv")
    {
        this->timestamp = ts ? ts : "";
        this->device_name = device_name ? device_name : "";
        this->framework = fw ? fw : "";
        this->workload_name = w_name ? w_name : "";
        this->workload_type = w_type ? w_type : "";
        this->device = dev ? dev : "";
        this->batch_size = b_size;
        this->input_size = in_size;
        this->optimizer = opt ? opt : "";
        this->learning_rate = lr;
        this->beta1 = b1;
        this->beta2 = b2;
        this->epsilon = eps;
        this->time_ms = t_ms;
        this->batch_index = b_index;
        this->loss = l;
        this->accuracy = acc;
        this->log_filename = log_file ? log_file : "benchmarks-logs.csv";
    }

    /**
     * @brief Destroys the benchmark data object.
     */
    ~BenchmarkData()
    {
    }

    /**
     * @brief Returns the CSV header matching the serialized field order.
     * @return CSV header string.
     */
    static constexpr const char *csv_header()
    {
        return "timestamp,device_name,framework,workload_name,workload_type,device,batch_size,input_size,optimizer,learning_rate,beta1,beta2,epsilon,time_ms,batch_index,loss,accuracy";
    }

    /**
     * @brief Serializes the benchmark data into a CSV row.
     * @return CSV-formatted data row.
     */
    std::string to_csv_string() const
    {
        std::stringstream ss;
        ss << timestamp << "," << device_name << "," << framework << "," << workload_name << ","
           << workload_type << "," << device << "," << batch_size << ","
           << input_size << "," << optimizer << ","
           << learning_rate << "," << beta1 << "," << beta2 << ","
           << epsilon << "," << time_ms << "," << batch_index << "," << loss << "," << accuracy;
        return ss.str();
    }
};
