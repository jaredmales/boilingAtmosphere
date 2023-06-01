
//Make sure this switch is propagating
#ifndef EIGEN_NO_CUDA
#define EIGEN_NO_CUDA
#endif

//Make sure the PacketMathHalf.h isn't included b/c it appears broken.
#define EIGEN_PACKET_MATH_HALF_CUDA_H

#ifndef boilingAtmosphere_hpp
#define boilingAtmosphere_hpp


#include <boost/math/constants/constants.hpp>

#include <mx/math/func/jinc.hpp>
#include <mx/sigproc/psdUtils.hpp>
#include <mx/improc/eigenImage.hpp>
#include <mx/ioutils/fileUtils.hpp>
#include <mx/ioutils/fits/fitsFile.hpp>

#include <mx/ao/analysis/aoSystem.hpp>


#ifndef BOILATM_NO_CUDA
#include <mx/math/cuda/templateCublas.hpp>
#include <mx/math/cuda/templateCufft.hpp>
#include <mx/math/cuda/templateCurand.hpp>
#include <mx/math/cuda/cudaPtr.hpp>
#else
#include <mx/math/randomT.hpp>
#include <mx/math/fft/fft.hpp>
#include <mx/math/fft/fftwEnvironment.hpp>
#endif

#include <mx/mxException.hpp>
#include <mx/app/application.hpp>

namespace mx 
{
namespace AO 
{
namespace analysis
{

template<typename _realT>
class boilingAtmosphere
{

public:
   typedef _realT realT;

#ifndef BOILATM_NO_CUDA
   typedef typename mx::cuda::complex<realT>::cudaType complexT;
#else
   typedef typename std::complex<realT> complexT;
#endif
   
protected:
   std::string m_dir {"data"}; ///< Directory for output files 

   size_t m_scrnSz {1024}; ///< Size, in pixels, of the turbulence screens.

   size_t m_wfSz {128}; ///< Size, in pixels, of the wavefronts to be extracted from the screens.
   
   aoSystem<realT, vonKarmanSpectrum<realT>> * m_aosys;

   bool m_ownAOSys {false}; ///< flag indicating whether or not the m_aosys pointer is owned.  It is de-allocated if so.

   realT m_fs {1000}; ///< the sampling frequency, in Hz.

   std::vector<realT> m_alpha1s; ///< The alpha_1 parameters for each layer.

   bool m_pureFF {true}; ///< Whether or not to use pure frozen flow.  Automatically set to false if alpha_1 is set.
   
   bool m_scaleAlpha {true}; ///< Whether or not to scale alphas by eddy size.  

   uint64_t m_seed {0}; ///< The random number generator seed.  If 0, then the seed is created.

#ifndef BOILATM_NO_CUDA
   curandGenerator_t m_gen; ///< CUDA random generator handle.
      
   cufftHandle m_fftPlan {0}; ///< CUDA FFT plan handle.  
   
   cublasHandle_t m_handle;
    
   std::vector<mx::cuda::cudaPtr<realT>> m_psdSqrt; //Each layer gets it's own PSD.

   
   
   std::vector<mx::cuda::cudaPtr<realT>> m_alphas;  ///< The auto-regressive parameter for each layer
   std::vector<mx::cuda::cudaPtr<realT>> m_alphas1M; ///< One minus  the AR-1 parameter for each layer
   
   std::vector<mx::cuda::cudaPtr<std::complex<realT>>> m_windPhase; ///< The phase shift screen for each layer.
   
   
   std::vector<mx::cuda::cudaPtr<std::complex<realT>>> m_transPhase; //The transformed phase for each layer.
   
   mx::cuda::cudaPtr<std::complex<realT>> m_noise; ///< Working memory for phase calcuation.
   
   mx::cuda::cudaPtr<realT> m_phase;

#else

   std::vector<mx::math::normDistT<realT>> normVar;
   
   std::vector<mx::improc::eigenImage<std::complex<realT>>> m_alphas;  ///< The auto-regressive parameter for each layer
   std::vector<mx::improc::eigenImage<std::complex<realT>>> m_alphas1M; ///< One minus  the AR-1 parameter for each layer
   
   mx::math::fft::fftT< complexT, complexT,2,0> fft_fwd; ///< FFT object for the forward transform.
   mx::math::fft::fftT< complexT, complexT,2,0> fft_inv; ///< FFT object for the inverse transform.
   
   std::vector<mx::improc::eigenImage<std::complex<realT>>> m_psdSqrt; //Each layer gets it's own PSD.

   std::vector<mx::improc::eigenImage<std::complex<realT>>> m_windPhase; 
   
   std::vector<mx::improc::eigenImage<std::complex<realT>>> m_transPhase; //The transformed phase for each layer.
   
   std::vector<mx::improc::eigenImage<std::complex<realT>>> m_noise; ///< Working memory for phase calcualtion.
   
   mx::improc::eigenImage<realT> m_phase;

#endif

public:   

   boilingAtmosphere();

   ~boilingAtmosphere();

   void dir( const std::string & d );

   std::string dir();

   void scrnSz( const size_t & sSz );

   size_t scrnSz();

   void wfSz( const size_t & wSz );

   size_t wfSz();

   void aosys( aoSystem<realT, vonKarmanSpectrum<realT>> * aosys );

   void allocAOSys();

   void deallocAOSys();

   aoSystem<realT, vonKarmanSpectrum<realT>> * aosys( );

   realT fs();

   void alpha1( const realT & alph1 );
   
   void alpha1s( const std::vector<realT> & alph1s );

   void scaleAlpha( const bool & sa );

   bool scaleAlpha();

   void pureFF( const bool & pff );

   bool pureFF();

   int allocate();

   
   /// Generate a single layer
   /** The result is stored in m_noise.  Not thread safe.
     *
     * \returns 0 on success
     */ 
   int genLayer( const size_t & n /**< [in] the layer number */);
   
   /// Generate a fourier domain phase screen for each layer
   /** The result is stored in m_transPhase.  Works by calling genlayer(const size_t &) for 
     * each layer.
     *
     * \returns 0 on succcess
     */ 
   int genLayers();
   
   int updateLayers();
   
   int getWavefront( mx::improc::eigenImage<realT> & wf);

   ///Output current parameters to a stream
   /** Outputs a formatted list of all current parameters.
     *
     */ 
   template<typename iosT>
   iosT & dumpBoilingAtmosphere( iosT & ios /**< [in] a std::ostream-like stream. */);

   /// \name mx::application support
   /** @{
     */
   
   void setupConfig( mx::app::appConfigurator & config );

   void loadConfig( mx::app::appConfigurator & config );


   /// @}

};

template<typename realT>
void boilingAtmosphere<realT>::dir( const std::string & d )
{
   m_dir = d;
}

template<typename realT>
std::string boilingAtmosphere<realT>::dir()
{
   return m_dir;
}

template<typename realT>
void boilingAtmosphere<realT>::scrnSz( const size_t & sSz )
{
   m_scrnSz = sSz;
}

template<typename realT>
size_t boilingAtmosphere<realT>::scrnSz()
{
   return m_scrnSz;
}

template<typename realT>
void boilingAtmosphere<realT>::wfSz( const size_t & wSz )
{
   m_wfSz = wSz;
}

template<typename realT>
size_t boilingAtmosphere<realT>::wfSz()
{
   return m_wfSz;
}

template<typename realT>
void boilingAtmosphere<realT>::aosys( aoSystem<realT, vonKarmanSpectrum<realT>> * aosys )
{
   if(aosys == nullptr) return allocAOSys();

   if(m_aosys && m_ownAOSys)
   {
      delete m_aosys;
   }

   m_aosys = aosys;
   m_ownAOSys = false;
}

template<typename realT>
void boilingAtmosphere<realT>::allocAOSys()
{
   if(m_aosys && m_ownAOSys)
   {
      delete m_aosys;
   }

   m_aosys = new aoSystem<realT, vonKarmanSpectrum<realT>>;
   m_ownAOSys = true;
}

template<typename realT>
void boilingAtmosphere<realT>::deallocAOSys()
{
   if(m_aosys && m_ownAOSys)
   {
      delete m_aosys;
   }

   m_aosys = nullptr;
   m_ownAOSys = false;
}

template<typename realT>
aoSystem<realT, vonKarmanSpectrum<realT>> * boilingAtmosphere<realT>::aosys( )
{
   return m_aosys;
}

template<typename realT>
realT boilingAtmosphere<realT>::fs()
{
   return m_fs;
}

template<typename realT>
void boilingAtmosphere<realT>::alpha1( const realT & alph1)
{
   if(m_aosys == nullptr)
   {
      mxThrowException( err::paramnotset, "boilingAtmosphere::alpha1", "AO system m_aosys is not set or allocated" );
   }
   
   alpha1s( std::vector<realT>( m_aosys->atm.n_layers(), alph1));
}

template<typename realT>
void boilingAtmosphere<realT>::alpha1s( const std::vector<realT> & alph1s)
{
   if(m_aosys == nullptr)
   {
      mxThrowException( err::paramnotset, "boilingAtmosphere::alpha1s", "AO system m_aosys is not set or allocated" );
   }
   
   if(alph1s.size() != m_aosys->atm.n_layers())
   {
      mxThrowException( err::sizeerr, "boilingAtmosphere::alpha1s", "input vector size does not match number of layers in atmosphere" );
   }
      
   m_alpha1s = alph1s;
   
   m_pureFF = false;
}

template<typename realT>
void boilingAtmosphere<realT>::scaleAlpha( const bool & sa )
{
   m_scaleAlpha = sa;
}

template<typename realT>
bool boilingAtmosphere<realT>::scaleAlpha()
{
   return m_scaleAlpha;
}

template<typename realT>
void boilingAtmosphere<realT>::pureFF( const bool & pff )
{
   m_pureFF = pff;
}

template<typename realT>
bool boilingAtmosphere<realT>::pureFF()
{
   return m_pureFF;
}

#ifndef BOILATM_NO_CUDA
#include "boilingAtmosphereCuda.cpp"
#else
#include "boilingAtmosphereNoCuda.cpp"
#endif


template<typename realT>
template<typename iosT>
iosT & boilingAtmosphere<realT>::dumpBoilingAtmosphere( iosT & ios)
{
   if(m_aosys) m_aosys->dumpAOSystem(ios);

   ios << "# Boiling Atmosphere Params:\n";
   ios << "#    dir = " << dir() << '\n';
   ios << "#    scrnSz = " << scrnSz() << '\n';
   ios << "#    wfSz = " << wfSz() << '\n';
   ios << "#    fs = " << fs() << '\n';
   ios << "#    tau_1 = ";
   for(size_t n=0; n < m_alpha1s.size()-1; ++n) ios << -0.001/log(m_alpha1s[n])<< ", ";
   if(m_alpha1s.size() > 0) ios << -0.001/log(m_alpha1s[m_alpha1s.size()-1]);
   ios << '\n';

   ios << "#    alpha = ";
   for(size_t n=0; n < m_alpha1s.size()-1; ++n) ios << m_alpha1s[n] << ", ";
   if(m_alpha1s.size() > 0) ios << m_alpha1s[m_alpha1s.size()-1];
   ios << '\n';

   ios << "#    pureFF = " << std::boolalpha << m_pureFF << '\n';
   ios << "#    scaleAlpha = " << std::boolalpha << m_scaleAlpha << '\n';
   ios << "#    seed = " << m_seed << '\n';   

   return ios;

}

template<typename realT>
void boilingAtmosphere<realT>::setupConfig( mx::app::appConfigurator & config )
{
   using namespace mx::app;

   if(!m_aosys) allocAOSys();

   m_aosys->setupConfig(config);

   config.add("boilatm.directory","", "boilatm.directory",argType::Required, "boilatm", "directory", false, "string", "Directory for output files");
   config.add("boilatm.screenSz","", "boilatm.screenSz",argType::Required, "boilatm", "screenSz", false, "int", "Size of the phase screens [pix] (default: 1024)");
   config.add("boilatm.wavefrontSz","", "boilatm.wavefrontSz",argType::Required, "boilatm", "wavefrontSz", false, "int", "Size of the wavefront [pix] (<= screenSz) (default: 128)");
   config.add("boilatm.alpha1","", "boilatm.alpha1",argType::Required, "boilatm", "alpha1", false, "vector<real>", "The boiling AR coefficient.  Per layer, if one is specified it applies to all layers. Overrides tau1");
   config.add("boilatm.tau1","", "boilatm.tau11",argType::Required, "boilatm", "tau1", false, "vector<real>", "The boiling timescale.  Per layer, if one is specified it applies to all layers. (default: 0.5)");
   config.add("boilatm.scale","", "boilatm.scale",argType::Required, "boilatm", "scale", false, "bool", "Whether or not to scale alpha. (default: true)");
   config.add("boilatm.pureFF","", "boilatm.pureFF",argType::Required, "boilatm", "pureFF", false, "bool", "If true, then no boiling is applied. (default: false)");


}

template<typename realT>
void boilingAtmosphere<realT>::loadConfig( mx::app::appConfigurator & config )
{

   if(m_aosys) m_aosys->loadConfig(config);

   config(m_dir, "boilatm.directory");
   config(m_scrnSz, "boilatm.screenSz");
   config(m_wfSz, "boilatm.wavefrontSz");

   m_fs = 1.0/m_aosys->tauWFS();

   std::vector<realT> t1;
   config(t1, "boilatm.tau1");
   if(t1.size() == 1)
   {
      alpha1(exp(-0.001/t1[0]));
   }
   else if(t1.size() > 1)
   {
      for(size_t n=0 ; n < t1.size(); ++n)
      {
         t1[n] = exp(-0.001/t1[n]);
      }
      alpha1s(t1);
   }

   std::vector<realT> a1;
   config(a1,"boilatm.alpha1");
   if(a1.size() == 1)
   {
      alpha1(a1[0]);
   }
   else if(a1.size() > 1) alpha1s(a1);

   config(m_scaleAlpha, "boilatm.scale");
   config(m_pureFF, "boilatm.pureFF");

}

} //namespace analysis
} //namespace AO
} //namespace mx

#endif
