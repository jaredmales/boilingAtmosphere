//This is included in boilingAtmosphere.hpp when needed.

using namespace mx;

using namespace boost::math::constants;

template<typename realT>
boilingAtmosphere<realT>::boilingAtmosphere()
{
   curandCreateGenerator(&m_gen, CURAND_RNG_PSEUDO_DEFAULT);
   
   cublasCreate(&m_handle);
   //cublasSetPointerMode(m_handle, CUBLAS_POINTER_MODE_DEVICE);
}
template boilingAtmosphere<float>::boilingAtmosphere();
template boilingAtmosphere<double>::boilingAtmosphere();

template<typename realT>
boilingAtmosphere<realT>::~boilingAtmosphere()
{
   deallocAOSys();

   if( m_fftPlan )
   {
      cufftDestroy(m_fftPlan);
   }

}
template boilingAtmosphere<float>::~boilingAtmosphere();
template boilingAtmosphere<double>::~boilingAtmosphere();
   
   
template<typename realT>
int boilingAtmosphere<realT>::allocate()
{
   if(m_aosys == nullptr)
   {
      mxThrowException( err::paramnotset, "boilingAtmosphere::allocate", "AO System m_aosys is not set or allocated" );
   }

   ioutils::createDirectories(m_dir);

   m_psdSqrt.resize( m_aosys->atm.n_layers() );
   
   m_windPhase.resize( m_aosys->atm.n_layers() );
   
   m_alphas.resize( m_aosys->atm.n_layers() );
   
   m_alphas1M.resize( m_aosys->atm.n_layers() );

   fits::fitsFile<realT> ff;

   //Make frequency and psd arrays on host then upload.
   improc::eigenImage<realT> freq, k_x, k_y;
   
   freq.resize(m_scrnSz, m_scrnSz);
   sigproc::frequencyGrid(freq, m_aosys->D()/m_wfSz, k_x, k_y);

   ff.write("data/freq.fits", freq);
   improc::eigenImage<realT> psd;
   psd.resize(m_scrnSz, m_scrnSz);
   
   improc::eigenImage<std::complex<realT>> windPhase;
   windPhase.resize(m_scrnSz, m_scrnSz);
   
   improc::eigenImage<realT> alphas1M;
   if(!m_pureFF)
   {
      alphas1M.resize(m_scrnSz, m_scrnSz);
   }
   
   
   realT beta = 1.0/pow( m_aosys->D()/m_wfSz,2); //scales for spatial frequency sampling
   
   for(size_t n=0; n < m_aosys->atm.n_layers(); ++n)
   {   
      realT tau1;
      if(!m_pureFF)
      {
         tau1 = -0.001/log(m_alpha1s[n]);
      }
      
      for(size_t ii =0; ii < m_scrnSz; ++ii)
      {
         for(size_t jj=0; jj < m_scrnSz; ++jj)
         {
            realT tpsd = beta * m_aosys->psd(m_aosys->atm, n, freq(ii,jj), m_aosys->lam_sci(), m_aosys->lam_wfs(), m_aosys->secZeta());
            
            psd(ii,jj) = sqrt(tpsd)/pow(m_scrnSz,2); //m_scrnSz^2 is for FFT norm.
                        
            realT vx = m_aosys->atm.layer_v_wind(n)*cos(m_aosys->atm.layer_dir(n));
            realT vy = m_aosys->atm.layer_v_wind(n)*sin(m_aosys->atm.layer_dir(n));
            
            std::complex<realT> phase = -two_pi<realT>()/m_fs*(k_x(ii,jj)*vx+k_y(ii,jj)*vy) * std::complex<realT>(0,1);
            windPhase(ii,jj) = exp(phase);
            
            realT alph;
            
            if(m_pureFF)
            {
               alph = 1;
            }
            else if(m_scaleAlpha)
            {
               alph = exp( -pow( freq(ii,jj), 2./3.)/ (m_fs*tau1));
               
            }
            else
            {
               alph = m_alpha1s[n];
            }

            windPhase(ii,jj) *= alph;
            
            if(!m_pureFF)
            {
               alphas1M(ii,jj) = sqrt(1.0-pow(alph,2));
            }
         }
      }
      
      
      m_psdSqrt[n].upload(psd.data(), m_scrnSz*m_scrnSz);///\todo error check
      m_windPhase[n].upload(windPhase.data(), m_scrnSz*m_scrnSz);
      
      if(!m_pureFF)
      {
         m_alphas1M[n].upload(alphas1M.data(), m_scrnSz*m_scrnSz);// = sqrt(1-pow(alphas[n],2));
      
         ff.write("data/alphas1M_" + std::to_string(n) + ".fits", alphas1M);
      }
            
   }
   
   
   m_transPhase.resize( m_aosys->atm.n_layers() );///\todo error check
   
   for(size_t n=0; n< m_transPhase.size(); ++n) 
   {
      m_transPhase[n].resize(m_scrnSz*m_scrnSz);///\todo error check
   }
   
   m_noise.resize(m_scrnSz*m_scrnSz);///\todo error check
   
   m_phase.resize(m_scrnSz*m_scrnSz);
   
   mx::cuda::cufftPlan2d<complexT,complexT>(&m_fftPlan, m_scrnSz, m_scrnSz);///\todo error check
   
   if(m_seed == 0)
   {
      m_seed = time(0);
   }

   curandStatus_t rv = curandSetPseudoRandomGeneratorSeed(m_gen, m_seed);
   
   if(rv != CURAND_STATUS_SUCCESS)
   {
      std::cerr << "seeding failed\n";
   }

      
   return 0;
}

template int boilingAtmosphere<float>::allocate();

template int boilingAtmosphere<double>::allocate();



template<typename realT>
int boilingAtmosphere<realT>::genLayer( const size_t & n)
{
   mx::cuda::curandGenerateNormal<realT>( m_gen, (realT *) m_noise.m_devicePtr, m_noise.m_size*2, 0.0, 1.0); 
   
   mx::cuda::cufftExec<complexT, complexT>(m_fftPlan, (complexT *) m_noise.m_devicePtr, (complexT *) m_noise.m_devicePtr, CUFFT_FORWARD);    
      
   //Apply the filter.
   mx::cuda::elementwiseXxY( m_noise(), m_psdSqrt[n](), m_scrnSz*m_scrnSz);
   
   return 0;
}

template int boilingAtmosphere<float>::genLayer(const size_t & n);
template int boilingAtmosphere<double>::genLayer(const size_t & n);

template<typename realT>
int boilingAtmosphere<realT>::genLayers()
{
   
   for(size_t n=0; n<m_transPhase.size(); ++n)
   {

      genLayer(n);   
      
      cudaMemcpy(m_transPhase[n].m_devicePtr, m_noise.m_devicePtr, m_noise.m_size*sizeof(complexT), cudaMemcpyDeviceToDevice);
   
   }
   
   return 0;
}

template int boilingAtmosphere<float>::genLayers();
template int boilingAtmosphere<double>::genLayers();

template<typename realT>
int boilingAtmosphere<realT>::updateLayers()
{
   
   //0) Zero out the phase screen
   realT zero = 0;
   mx::cuda::cublasTscal<realT>(m_handle, m_scrnSz*m_scrnSz, &zero, m_phase.m_devicePtr, 1);
   
   for(size_t n=0; n<m_transPhase.size(); ++n)
   {
      //1) Propagate by frozen flow, including the boiling alpha
      mx::cuda::elementwiseXxY(m_transPhase[n](), m_windPhase[n](), m_scrnSz*m_scrnSz);
         
      if(!m_pureFF)
      {
         //2) Generate new screen for this layer in the Fourier domain.
         genLayer(n);
   
         //3) Multiply new layer by 1-alpha in the Fourier domain, and add it to the existing layer.
         mx::cuda::elementwiseXxY(m_noise(), m_alphas1M[n](), m_scrnSz*m_scrnSz);
      
         complexT one = {1,0};
         mx::cuda::cublasTaxpy<complexT>(m_handle, m_scrnSz*m_scrnSz, &one, (complexT *) m_noise.m_devicePtr, 1, (complexT *) m_transPhase[n].m_devicePtr, 1);
      }
      //4) Now Fourier transform to the space domain
      mx::cuda::cufftExec<complexT, complexT>(m_fftPlan, (complexT *) m_transPhase[n].m_devicePtr, (complexT *) m_noise.m_devicePtr, CUFFT_INVERSE);

      //5) And accumulate the real part as the new phase screen, weighted by Cn2.
      realT Cn2 = sqrt(m_aosys->atm.layer_Cn2(n));
      mx::cuda::cublasTaxpy<realT>(m_handle, m_scrnSz*m_scrnSz, &Cn2, (realT *) m_noise.m_devicePtr, 2, m_phase.m_devicePtr , 1);
      
   }
   
   return 0;
}

template int boilingAtmosphere<float>::updateLayers();
template int boilingAtmosphere<double>::updateLayers();

template<typename realT>
int boilingAtmosphere<realT>::getWavefront( mx::improc::eigenImage<realT> & wf)
{
   //wf.resize( m_wfSz, m_wfSz);
      
   cudaMemcpy2D( wf.data(), m_wfSz*sizeof(realT), m_phase.m_devicePtr,  m_scrnSz*sizeof(realT), m_wfSz*sizeof(realT), m_wfSz, cudaMemcpyDeviceToHost);
   
   return 0;
}

template int boilingAtmosphere<float>::getWavefront(mx::improc::eigenImage<float> & wf);
template int boilingAtmosphere<double>::getWavefront(mx::improc::eigenImage<double> & wf);
