#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>


#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "matrix.h"
//#include "io.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;
using std::tie;
using std::make_tuple;
using std::tuple;
using std::get;

using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

#define cellGridHOG 8
#define cellGridCOL 8
#define cellGridLBP 4
#define segmentCount 12


// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'

Matrix<double> custom(Matrix<double> src_image, Matrix<double> kernel) {

    //if(mirrorFlag) src_image = mirror(src_image, kernel.n_rows, kernel.n_cols);

    Matrix<double> tempImage(src_image.n_rows, src_image.n_cols);
    for(int x =0; x < src_image.n_cols; x++ )
    {
        for(int y =0; y < src_image.n_rows; y++ )
        {
            tempImage(y,x) = 0;
        }
    }

    for(int x = kernel.n_cols/2; x < src_image.n_cols-kernel.n_cols/2; x++ )
    {
       for(int y = kernel.n_rows/2; y < src_image.n_rows - kernel.n_rows/2; y++ )
        {
            for (int dx = -kernel.n_cols/2, kx=0; kx < kernel.n_cols; kx++, dx++)
            {
                for (int dy = -kernel.n_rows/2, ky=0; ky < kernel.n_rows; ky++, dy++)
                {                    
                    tempImage(y,x)+= src_image(y+dy,x+dx)*kernel(ky,kx);
                }
            }
        }
    }    
    
    //fixOverflow(tempImage);
    //if(mirrorFlag) tempImage = tempImage.submatrix(kernel.n_rows/2,kernel.n_cols/2,tempImage.n_rows -kernel.n_rows+1,tempImage.n_cols -kernel.n_cols+1 );
    return tempImage;
}

void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t dir_idx = 0; dir_idx < file_list.size(); ++dir_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[dir_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[dir_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels, 
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}
/*
void fixOverflow(Image &src_image)
{
    for(int x =0; x < src_image.n_cols; x++ )
    {
        for(int y =0; y < src_image.n_rows; y++ )
        {
            get<0>( src_image(y,x) )= fmax( 0 , fmin(255, get<0>( src_image(y,x) ) ) );
            get<1>( src_image(y,x) )= fmax( 0 , fmin(255, get<1>( src_image(y,x) ) ) );
            get<2>( src_image(y,x) )= fmax( 0 , fmin(255, get<2>( src_image(y,x) ) ) );
        }
    }
}
*/
// Exatract features from dataset.
// You should implement this function by yourself =)
void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
    int n,m;
    double PI = atan2(0,-1);
    
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {

        vector<float> one_image_features;
        vector<float> HOG;
        vector<float> COL;
        vector<float> LBP;
        n=data_set[image_idx].first->TellHeight();
        m=data_set[image_idx].first->TellWidth();
        Matrix<double> gray( n, m);
        Matrix<double> dir( n, m);
        Matrix<double> val( n, m);
        Matrix<double> sobY( n, m);
        Matrix<double> sobX( n, m);
        //Image debugImg( n, m);
        //Image debugImg2( n, m);
        
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                RGBApixel pixel = data_set[image_idx].first->GetPixel(j,i);                 
                gray(i,j) =  0.299 * pixel.Red +  0.587* pixel.Blue +  0.114* pixel.Green;
               // debugImg(i,j) = make_tuple(dir(i,j), dir(i,j), dir(i,j) );

            }
        }
        //cout<<"\n"<<(int)image_idx;

        Matrix<double> kernel = {{-1, 0, 1},
                             {-2, 0, 2},
                             {-1, 0, 1}};
        sobX = custom(gray,kernel);

        kernel = {{ -1,  -2,  -1},
                 { 0,  0,  0},
                 {1, 2, 1}};
        sobY = custom(gray,kernel);

        for(int x =0; x < m; x++ )
        {
            for(int y =0; y < n; y++ )
            {
                val(y,x)= sqrt( sobX(y,x) * sobX(y,x) + sobY(y,x) * sobY(y,x)  );
                dir(y,x) = atan2(sobY(y,x), sobX(y,x) );
                /*
                get<0>( debugImg(y,x) )=  (PI + dir(y,x))*255/2/PI; //sqrt( sobX(y,x) * sobX(y,x) + sobY(y,x) * sobY(y,x)  );
                get<1>( debugImg(y,x) )= get<0>( debugImg(y,x) );
                get<2>( debugImg(y,x) )= get<0>( debugImg(y,x) );
                get<0>( debugImg2(y,x) )= val(y,x);
                get<1>( debugImg2(y,x) )= get<0>( debugImg2(y,x) );
                get<2>( debugImg2(y,x) )= get<0>( debugImg2(y,x) );
                */
            }
        }
      //  fixOverflow(debugImg);
     //   fixOverflow(debugImg2);
     //   save_image(debugImg, "1.bmp");
     //   save_image(debugImg2, "2.bmp");

        

        for (int xy = 0; xy < cellGridHOG*cellGridHOG; ++xy)
        {
            int dy = xy/cellGridHOG;
            int dx = xy%cellGridHOG;
            vector<float> H(segmentCount,0);
            
           

            for (int i = (1.0*dy/cellGridHOG)*n; i < (1.0*(dy+1)/cellGridHOG)*n; ++i)
            {
                for (int j = (1.0*dx/cellGridHOG)*m; j < (1.0*(dx+1)/cellGridHOG)*m; ++j)
                {
                    int seg = floor((dir(i,j)+PI)/PI/2*segmentCount);
                    H[seg]+= val(i,j);
                    
                    //if(xy==0)cout<<"WTF"<<i<<" "<<j<<" "<<seg<<" "<<dir(i,j)<<" "<<val(i,j)<<" "<<floor((dir(i,j)+PI)/PI/2*8)<<"\n";
                }
            }          

            
            float max = H[0];

            for (int i = 1; i < segmentCount; ++i)
                if( H[i]>max) max = H[i];

            if(max!=0)            
                for (int i = 0; i < segmentCount; ++i)
                    H[i]/=max;
            

            HOG.insert(HOG.end(), H.begin(), H.end() );

        }
        
       
        for (int xy = 0; xy < cellGridCOL*cellGridCOL; ++xy)
        {
            int dy = xy/cellGridCOL;
            int dx = xy%cellGridCOL;
            int pixNum=0;
            double R=0,G=0,B=0;
            vector<float> C(3,0);

            for (int i = (1.0*dy/cellGridCOL)*n; i < (1.0*(dy+1)/cellGridCOL)*n; ++i)
            {
                for (int j = (1.0*dx/cellGridCOL)*m; j < (1.0*(dx+1)/cellGridCOL)*m; ++j)
                {
                    
                    R+= data_set[image_idx].first->GetPixel(j,i).Red;
                    G+= data_set[image_idx].first->GetPixel(j,i).Green;
                    B+= data_set[image_idx].first->GetPixel(j,i).Blue;
                    pixNum++;
                    //if(xy==0)cout<<"WTF"<<i<<" "<<j<<" "<<seg<<" "<<dir(i,j)<<" "<<val(i,j)<<" "<<floor((dir(i,j)+PI)/PI/2*8)<<"\n";
                }
            }
            C[0] = R/pixNum/255;
            C[1] = G/pixNum/255;
            C[2] = B/pixNum/255;

            COL.insert(COL.end(), C.begin(), C.end() );
            
        }

        n-=2;
        m-=2;

        for (int xy = 0; xy < cellGridLBP*cellGridLBP; ++xy)
        {
            int dy = xy/cellGridLBP;
            int dx = xy%cellGridLBP;
            
            
            vector<float> L(256,0);
            int pattern = 0;

            for (int i = (1.0*dy/cellGridLBP)*n+1; i < (1.0*(dy+1)/cellGridLBP)*n+1; ++i)
            {
                for (int j = (1.0*dx/cellGridLBP)*m+1; j < (1.0*(dx+1)/cellGridLBP)*m+1; ++j)
                {
                    pattern=0;
                    if( gray(i,j) <= gray(i-1,j) ) pattern+=1;
                    if( gray(i,j) <= gray(i-1,j+1) ) pattern+=2;
                    if( gray(i,j) <= gray(i,j+1) ) pattern+=4;
                    if( gray(i,j) <= gray(i+1,j+1) ) pattern+=8;

                    if( gray(i,j) <= gray(i+1,j) ) pattern+=16;
                    if( gray(i,j) <= gray(i+1,j-1) ) pattern+=32;
                    if( gray(i,j) <= gray(i,j-1) ) pattern+=64;
                    if( gray(i,j) <= gray(i-1,j-1) ) pattern+=128;
                    L[pattern]++;
                }
            }
            int max =0;

            for (int i = 0; i < 256; ++i)            
                if(L[i]>max) max=L[i];

            if(max!=0)
                for (int i = 0; i < 256; ++i) 
                    L[i]/=max;



            LBP.insert(LBP.end(), L.begin(), L.end() );
            
        }
        
        n+=2;
        m+=2;
        
        one_image_features.insert(one_image_features.end(), HOG.begin(), HOG.end() );
        one_image_features.insert(one_image_features.end(), COL.begin(), COL.end() );
        one_image_features.insert(one_image_features.end(), LBP.begin(), LBP.end() );
        /*
        if(image_idx==0)
        {
            cout <<one_image_features.size()<<"\n";
                  
            
            for (int i = 0; i < one_image_features.size(); ++i)
            {
                if(i<HOG.size() && i%segmentCount==0)cout << "\n";
                if(i>=HOG.size() && i<HOG.size() + COL.size() &&  (i-HOG.size())%3==0)cout << "\n";
                if(i>=HOG.size() + COL.size() && (i-HOG.size() - COL.size())%256==0)cout << "\n\n";
                cout<<one_image_features[i]<<" ";
            }
            
        }
        fflush(NULL);
        */
       
        features->push_back(make_pair(one_image_features, data_set[image_idx].second));
        
       

    }
}

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}



// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;
    
        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.01;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}