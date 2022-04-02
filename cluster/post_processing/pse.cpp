//
//  pse
//  Created by zhoujun on 11/9/19.
//  Copyright © 2019年 zhoujun. All rights reserved.
//
#include <queue>
#include <functional>
#include <math.h>
#include <map>
#include <algorithm>
#include <vector>
#include "include/pybind11/pybind11.h"
#include "include/pybind11/numpy.h"
#include "include/pybind11/stl.h"
#include "include/pybind11/stl_bind.h"

using namespace std;
namespace py = pybind11;


namespace pan{

    py::array_t<int32_t> pse(
    py::array_t<int32_t, py::array::c_style> text,
    py::array_t<float, py::array::c_style> similarity_vectors,
    py::array_t<int32_t, py::array::c_style> label_map,
    int32_t kernel_label_num,
    int32_t border_label_num,
    float dis_threshold = 0.8)
    {
        auto pbuf_text = text.request();
        auto pbuf_similarity_vectors = similarity_vectors.request();
        auto pbuf_label_map = label_map.request();
        if (pbuf_label_map.ndim != 2 || pbuf_label_map.shape[0]==0 || pbuf_label_map.shape[1]==0)
            throw std::runtime_error("label map must have a shape of (h>0, w>0)");
        int h = pbuf_label_map.shape[0];
        int w = pbuf_label_map.shape[1];
        if (pbuf_similarity_vectors.ndim != 3 || pbuf_similarity_vectors.shape[0]!=h || pbuf_similarity_vectors.shape[1]!=w || pbuf_similarity_vectors.shape[2]!=4 ||
            pbuf_text.shape[0]!=h || pbuf_text.shape[1]!=w)
            throw std::runtime_error("similarity_vectors must have a shape of (h,w,4) and text must have a shape of (h,w,4)");

        //初始化结果
        auto res = py::array_t<int32_t>(pbuf_text.size);
        auto pbuf_res = res.request();

        // 获取 text similarity_vectors 和 label_map的指针
        auto ptr_label_map = static_cast<int32_t *>(pbuf_label_map.ptr);
        auto ptr_text = static_cast<int32_t *>(pbuf_text.ptr);
        auto ptr_similarity_vectors = static_cast<float *>(pbuf_similarity_vectors.ptr);
        auto ptr_res = static_cast<int32_t *>(pbuf_res.ptr);


        std::queue<std::tuple<int, int, int32_t>> bor;    //border

        // 计算各个kernel的similarity_vectors
        float kernel_vector[kernel_label_num][5] = {0};

        // kernel,border像素入队列
        for (int i = 0; i<h; i++)
        {
            auto p_label_map = ptr_label_map + i*w;
            auto p_similarity_vectors = ptr_similarity_vectors + i*w*4;
            auto p_tcl_map = ptr_text + i*w;
            auto p_res = ptr_res + i*w;

            for(int j = 0, k = 0; j<w && k < w * 4; j++,k+=4)
            {
                int32_t label = p_label_map[j];
                if (label>0)
                {
                    //std::cout<< "label "<< label<<std::endl;
                    kernel_vector[label][0] += p_similarity_vectors[k];
                    kernel_vector[label][1] += p_similarity_vectors[k+1];
                    kernel_vector[label][2] += p_similarity_vectors[k+2];
                    kernel_vector[label][3] += p_similarity_vectors[k+3];
                    kernel_vector[label][4] += 1;

                }

                int32_t value = p_tcl_map[j];
                if (value>0)
                {
                    bor.push(std::make_tuple(i, j, value));
                }
                p_res[j] = value;
            }
        }

        for(int i=0;i<kernel_label_num;i++)    //  get final average kernel embedding
        {
            for (int j=0;j<4;j++)
            {
                kernel_vector[i][j] /= kernel_vector[i][4];
            }

        }

        // validation code
        double measure=0;
        double sum=0;
        // calulate kernel distance
        for(int i=1; i<kernel_label_num; i++)
        {
            for(int j=i+1; j<kernel_label_num;j++)
             {
                measure=0;
                for(int k=0; k<4; k++)
                {
                    measure+=pow(kernel_vector[i][k]-kernel_vector[j][k],2);
                }
                measure=sqrt(measure);
                //sum += measure;
                std::cout<<i<<"->"<<j<<" "<<measure<<std::endl;
            }
        }




       std::queue<std::tuple<int, int, float,int, int32_t>> border_dis;     // (y,x,dis,k,int32_t)
        while(!bor.empty()){
        auto q_n = bor.front();
        bor.pop();
        int y = std::get<0>(q_n);
        int x = std::get<1>(q_n);
        int32_t l = std::get<2>(q_n);
        auto p_similarity_vectors = ptr_similarity_vectors + y * w * 4;
        for(int k=1;k<kernel_label_num;k++)
            {
                auto kernel_cv = kernel_vector[k];
                // 计算距离
                float dis = 0;
                for(size_t i=0;i<4;i++)
                {
                   dis += pow(kernel_cv[i] - p_similarity_vectors[x * 4 + i],2);
                }
               dis = sqrt(dis);
               border_dis.push(std::make_tuple(y, x, dis, k, l));
            }
        }


        float dis_matrix[border_label_num][kernel_label_num+1];
        for(int i=0;i<border_label_num;i++)
        {
            for (int j=0;j<=kernel_label_num;j++)
            {

                dis_matrix[i][j]=0;     //initialization
            }
        }


        while(!border_dis.empty()){
        //get each queue menber in border
        auto q_n = border_dis.front();
        border_dis.pop();

        int y = std::get<0>(q_n);
        int x = std::get<1>(q_n);
        float dis = std::get<2>(q_n);
        int32_t k = std::get<3>(q_n);
        int32_t l = std::get<4>(q_n);

        for(int i=1;i<border_label_num;i++)
        {
            for(int j=1;j<kernel_label_num;j++)
                {
                    if ((l==i) && (k==j))
                        {dis_matrix[i][j] += dis;
                        dis_matrix[i][kernel_label_num] += 1;
                        }
                }
        }
        }


       // get dis_matrix
       for(int i=0;i<border_label_num;i++)
        {
            for (int j=0;j<kernel_label_num;j++)
            {
                dis_matrix[i][j] /= dis_matrix[i][kernel_label_num];
                //cout<<dis_matrix[i][j]<<" ";
            }
        //cout<<endl;
        }
        //cout<<"****************"<<endl;

        int32_t index[border_label_num]={0};    //initialize index_array
        int32_t visit[border_label_num]={0};    // a border instance is occupied

        for (int j=1;j<kernel_label_num;j++)        //traverse by column
        {
            std::vector<std::pair<float,int32_t>>tup;
            for(int i=1;i<border_label_num;i++)
            {
                //cout<<dis_matrix[i][j]<<" ";
                if (visit[i]==0)
                {
                    tup.push_back(std::make_pair(dis_matrix[i][j],i));
                }
            }
            //cout<<endl;


            std::sort(tup.begin(),tup.end());

            if (tup.size()!=0)
            {
             float minum_dis = tup[0].first;
            int32_t minum_index =tup[0].second;
            float minum_dis1 = tup[1].first;
            int32_t minum_index1 =tup[1].second;

            if (minum_dis<dis_threshold)
            {
                index[minum_index] = j;
                visit[minum_index] = 1;
            }

            if (minum_dis1<dis_threshold)
            {
                index[minum_index1] = j;
                visit[minum_index1] = 1;
            }

            }

        }

         index[0]=0;
        for (int i = 0; i<h; i++)
        {
            auto p_label_map = ptr_text + i*w;      // boreder pixel
            auto p_res = ptr_res + i*w;

            for(int j = 0, k = 0; j<w && k < w * 4; j++,k+=4)
            {
                int32_t label = p_label_map[j];
                int32_t LABEL=index[(int)label];

                if(label>0)
                    p_res[j] = LABEL;
                else
                    p_res[j] = 0;
            }
        }


       return res;

    }





    std::map<int,std::vector<float>> get_points(
    py::array_t<int32_t, py::array::c_style> label_map,
    py::array_t<float, py::array::c_style> score_map,
    int label_num)
    {
        auto pbuf_label_map = label_map.request();
        auto pbuf_score_map = score_map.request();
        auto ptr_label_map = static_cast<int32_t *>(pbuf_label_map.ptr);
        auto ptr_score_map = static_cast<float *>(pbuf_score_map.ptr);
        int h = pbuf_label_map.shape[0];
        int w = pbuf_label_map.shape[1];

        std::map<int,std::vector<float>> point_dict;
        std::vector<std::vector<float>> point_vector;
        for(int i=0;i<label_num;i++)
        {
            std::vector<float> point;
            point.push_back(0);
            point.push_back(0);
            point_vector.push_back(point);
        }

        for (int i = 0; i<h; i++)
        {
            auto p_label_map = ptr_label_map + i*w;
            auto p_score_map = ptr_score_map + i*w;
            for(int j = 0; j<w; j++)
            {
                int32_t label = p_label_map[j];
                if(label==0)
                {
                    continue;
                }
                float score = p_score_map[j];
                point_vector[label][0] += score;
                point_vector[label][1] += 1;
                point_vector[label].push_back(j);
                point_vector[label].push_back(i);
            }
        }


        for(int i=0;i<label_num;i++)
        {
            if(point_vector[i].size() > 2)
            {
                point_vector[i][0] /= point_vector[i][1];   //get average score
                point_dict[i] = point_vector[i];
            }
        }
        return point_dict;
    }



    std::vector<int> get_num(
    py::array_t<int32_t, py::array::c_style> label_map,
    int label_num)
    {
        auto pbuf_label_map = label_map.request();
        auto ptr_label_map = static_cast<int32_t *>(pbuf_label_map.ptr);
        int h = pbuf_label_map.shape[0];
        int w = pbuf_label_map.shape[1];

        std::vector<int> point_vector;
        for(int i=0;i<label_num;i++)
        {
            point_vector.push_back(0);
        }
        for (int i = 0; i<h; i++)
        {
            auto p_label_map = ptr_label_map + i*w;
            for(int j = 0; j<w; j++)
            {
                int32_t label = p_label_map[j];
                if(label==0)
                {
                    continue;
                }
                point_vector[label] += 1;
            }
        }
        return point_vector;
    }
}


PYBIND11_MODULE(pse, m){
    m.def("pse_cpp", &pan::pse, " re-implementation pse algorithm(cpp)", py::arg("tcl_label_map"), py::arg("similarity_vectors"), py::arg("kernel_label_map"), py::arg("kernel_label_num"),py::arg("tcl_label_num"), py::arg("dis_threshold")=0.8);
    m.def("get_points", &pan::get_points, " re-implementation pse algorithm(cpp)", py::arg("label_map"), py::arg("score_map"), py::arg("label_num"));
    m.def("get_num", &pan::get_num, " re-implementation pse algorithm(cpp)", py::arg("label_map"), py::arg("label_num"));
}

