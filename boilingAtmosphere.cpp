
#ifndef EIGEN_NO_CUDA
#define EIGEN_NO_CUDA
#endif

#include <iostream>

#include <mx/app/application.hpp>
#include <mx/sys/timeUtils.hpp>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/imageTransforms.hpp>

#include "boilingAtmosphere.hpp"

template<typename _realT>
class boilingAtmosphereApp : public mx::app::application
{
public:
   typedef _realT realT;

protected:

   mx::AO::analysis::boilingAtmosphere<realT> m_batm;

   int m_N {120000};

public:

   boilingAtmosphereApp()
   {
      m_batm.allocAOSys();
      m_batm.aosys()->loadMagAOX();
      m_batm.aosys()->atm.loadLCO();
      //m_batm.aosys()->atm.r_0(0.16, 0.5e-6);
      //m_batm.aosys()->atm.L_0(25.0);

      m_batm.scrnSz(1024);   
      m_batm.wfSz(384);
      m_batm.aosys()->lam_sci(0.8e-6);
      m_batm.fs(1000);
      std::vector<realT> alphas( m_batm.aosys()->atm.n_layers(), 0.99626);
      m_batm.alpha1s(alphas);

   }

   virtual void setupConfig()
   {
      m_batm.setupConfig(config);

      config.add("boilatm.nsteps","", "boilatm.nsteps",mx::app::argType::Required, "boilatm", "nsteps", false, "int", "The number of steps to simulate");
   }
   
   virtual void loadConfig()
   {
      m_batm.loadConfig(config);

      config(m_N, "boilatm.nsteps");
   }
   
   virtual int execute()
   {
   
      double t0, t1;
      timespec ts;

      std::cerr << "allocating . . . ";

      t0 =  mx::sys::get_curr_time(ts);

      m_batm.allocate();
   
      
      mx::improc::eigenImage<realT> wf;
      
      mx::improc::eigenCube<realT> wfCube;
      wf.resize(m_batm.wfSz(),m_batm.wfSz());
      wfCube.resize(m_batm.wfSz(), m_batm.wfSz(), 500);
   
      t1 =  mx::sys::get_curr_time(ts);

      std::cerr << "complete (" << t1 - t0 << " sec)\n" << std::endl;

   

      std::cerr << "generating . . . ";
      
      t0 =  mx::sys::get_curr_time(ts);

      m_batm.genLayers();

      t1 =  mx::sys::get_curr_time(ts);

      std::cerr << "complete (" << t1 - t0 << " sec)\n" << std::endl;

      std::ofstream fout;
      fout.open( m_batm.dir() + "/params.txt");
      m_batm.dumpBoilingAtmosphere(fout);
      fout.close();

      std::cerr << "beginning" << std::endl;
   
      mx::fits::fitsFile<realT> ff;

      int cubeNo =0;
      int frameNo = 0;
   
      t0 = mx::sys::get_curr_time(ts);
      for(int n=0; n < m_N; ++n)   
      {  
         m_batm.updateLayers();
         m_batm.getWavefront(wf);
      
         wfCube.image(frameNo) = wf; 
         ++frameNo;
      
         if(frameNo == 500)
         {
            std::string fn = "data/cube_" + std::to_string(cubeNo) + ".fits";
            ff.write(fn, wfCube);
            ++cubeNo;
            frameNo = 0;
            std::cerr << cubeNo << " " << n << "/" << m_N << "\n";
         }

      }
      t1 = mx::sys::get_curr_time(ts);
   
      std::cerr << (t1-t0)/m_N << "\n";
   
      return 0;

   }
};

int main( int argc,
          char ** argv
        )
{
   typedef double realT;

   boilingAtmosphereApp<realT> bap;
   return bap.main(argc, argv);
/*
   mx::AO::analysis::boilingAtmosphere<realT> batm;

   batm.allocAOSys();
   //mx::AO::analysis::aoAtmosphere<realT> atm;
   batm.aosys()->D(6.5);
   batm.aosys()->atm.loadLCO();
   batm.aosys()->atm.r_0(0.16, 0.5e-6);
   batm.aosys()->atm.L_0(25.0);
   
   //atm.v_wind(9.4);2048
   //atm.r_0(0.2, 0.5e-6);
   
   //atm.setSingleLayer(0.17, 0.5e-6, 25, 0, 10000., 10.0, pi<realT>()/5.5);
   
   
   //mkdir("data", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
   //mx::ioutils::createDirectories("data");
   
   batm.scrnSz(200);   
   batm.wfSz(50);
   batm.aosys()->lam_sci(0.8e-6);
   batm.fs(1000);
   //batm.atm(&atm);
   
   //batm.m_pureFF = true; 
   std::vector<realT> alphas( batm.aosys()->atm.n_layers(), 0.99626);
   batm.alpha1s(alphas);
   //batm.m_scaleAlpha = true;
   //batm.m_scaleAlpha = false;

   batm.pureFF(true);
   
   batm.allocate();
   
   std::cerr << "alloc complete" << std::endl;
   
   mx::improc::eigenImage<realT> wf;
      
   mx::improc::eigenCube<realT> wfCube;
   wf.resize(batm.wfSz(),batm.wfSz());
   wfCube.resize(batm.wfSz(), batm.wfSz(), 500);
   
   double t0, t1;
   
   int N=500;

   std::cerr << "generating" << std::endl;
   
   batm.genLayers();
   
   std::cerr << "beginning" << std::endl;
   
   mx::fits::fitsFile<realT> ff;

   int cubeNo =0;
   int frameNo = 0;
   timespec ts;
   t0 = mx::sys::get_curr_time(ts);
   for(int n=0; n < N; ++n)   
   {  
      batm.updateLayers();
      batm.getWavefront(wf);
      
      wfCube.image(frameNo) = wf; 
      ++frameNo;
      
      if(frameNo == 500)
      {
         std::string fn = "data/cube_" + std::to_string(cubeNo) + ".fits";
         ff.write(fn, wfCube);
         ++cubeNo;
         frameNo = 0;
         std::cerr << cubeNo << " " << n << "/" << N << "\n";
      }

      //ds9(wf);
   }
   t1 = mx::sys::get_curr_time(ts);
   
   std::cerr << (t1-t0)/N << "\n";
   
   return 0;
   */
}
