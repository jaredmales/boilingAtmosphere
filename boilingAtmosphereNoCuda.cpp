//This is included in boilingAtmosphere.hpp when needed.

using namespace mx;

using namespace boost::math::constants;

template<typename realT>
boilingAtmosphere<realT>::boilingAtmosphere()
{
}
template boilingAtmosphere<float>::boilingAtmosphere();
template boilingAtmosphere<double>::boilingAtmosphere();

template<typename realT>
boilingAtmosphere<realT>::~boilingAtmosphere()
{
   deallocAtm();
}
template boilingAtmosphere<float>::~boilingAtmosphere();
template boilingAtmosphere<double>::~boilingAtmosphere();
   
   
template<typename realT>
int boilingAtmosphere<realT>::allocate( realT pupD,
                                        size_t wfSz,
                                        size_t scrnSz,
                                        realT lambda,
                                        mx::AO::analysis::aoAtmosphere<realT> & atm
                                      )
{
   m_pupD = pupD;
   m_wfSz = wfSz;
   m_scrnSz = scrnSz;

   m_lambda = lambda;

   m_atm = &atm;
   
   m_psdSqrt.resize(m_atm->n_layers());
   
   m_windPhase.resize(m_atm->n_layers());
   
   //Make frequency and psd arrays on host then upload.
   improc::eigenImage<realT> freq;
   
   freq.resize(m_scrnSz, m_scrnSz);
   sigproc::frequency_grid(freq, m_pupD/m_wfSz);
   
   improc::eigenImage<std::complex<realT>> psd;
   psd.resize(scrnSz, scrnSz);
   
   improc::eigenImage<std::complex<realT>> windPhase;
   windPhase.resize(scrnSz, scrnSz);
   
   realT beta = 0.0218/pow( m_atm->r_0(m_lambda), 5./3.)/pow( m_pupD/m_wfSz,2) ;
   
   for(size_t n=0; n < m_atm->n_layers(); ++n)
   {
   
      realT sqrt_alpha = 0.5*11./3.;

      realT L02;
      
      if(m_atm->L_0(n) > 0) L02 = 1.0/pow( m_atm->L_0(n), 2);
      else L02 = 0;
      
      for(size_t ii =0; ii < m_scrnSz; ++ii)
      {
         for(size_t jj=0; jj < m_scrnSz; ++jj)
         {
            
            realT Ppiston = pow(2*math::func::jinc(pi<realT>() * freq(ii,jj) * m_pupD), 2);
            
            realT tpsd;
            
            tpsd = beta / pow( pow(freq(ii,jj),2) + L02, sqrt_alpha);
            //if(m_atm->l_0(n) > 0 ) tpsd *= exp(-1*pow( freq(ii,jj)*m_atm->l_0(n), 2));
         
            tpsd *= (1 - Ppiston);
            
            
            psd(ii,jj) = std::complex<realT>(sqrt(tpsd)/pow(m_scrnSz,2),0); //m_scrnSz^2 is for FFT norm.
            
            realT dk = mx::sigproc::freq_sampling(m_scrnSz, 0.5/( m_pupD/m_wfSz));
            realT kx = ii*dk;
            if(ii > 0.5*(m_scrnSz - 1)+1) kx = ((int)ii-(int) m_scrnSz)*dk;
            
            realT ky = jj*dk;
            if(jj > 0.5*(m_scrnSz - 1)+1) ky = ((int)jj-(int)m_scrnSz)*dk;
            
            realT vx = m_atm->layer_v_wind(n)*cos(m_atm->layer_dir(n));
            realT vy = m_atm->layer_v_wind(n)*sin(m_atm->layer_dir(n));
            
            std::complex<realT> phase = -two_pi<realT>()/m_fs*(kx*vx+ky*vy) * std::complex<realT>(0,1);
            windPhase(ii,jj) = exp(phase);
         }
      }

      m_psdSqrt[n] = psd; //.upload(psd.data(), m_scrnSz*m_scrnSz);///\todo error check
      
      m_windPhase[n] = windPhase; //.upload(windPhase.data(), m_scrnSz*m_scrnSz);
      
   }
   
   m_transPhase.resize( m_atm->n_layers() );
   m_noise.resize( m_atm->n_layers() );
   normVar.resize( m_atm->n_layers() );
   for(size_t n=0; n< m_transPhase.size(); ++n) 
   {
      m_transPhase[n].resize(m_scrnSz,m_scrnSz);
      m_noise[n].resize(m_scrnSz, m_scrnSz);
      normVar[n].seed();

   }
   
   m_phase.resize(m_scrnSz, m_scrnSz);
   
   fft_fwd.plan(m_scrnSz, m_scrnSz, MXFFT_FORWARD, true);
   fft_inv.plan(m_scrnSz, m_scrnSz, MXFFT_BACKWARD, false);
   
   return 0;
}

template int boilingAtmosphere<float>::allocate( float pupD,
                                                 size_t wfSz,
                                                 size_t scrnSz,
                                                 float lambda,
                                                 mx::AO::analysis::aoAtmosphere<float> & atm
                                               );

template int boilingAtmosphere<double>::allocate( double pupD,
                                                  size_t wfSz,
                                                  size_t scrnSz,
                                                  double lambda,
                                                  mx::AO::analysis::aoAtmosphere<double> & atm
                                                );

template<typename realT>
int boilingAtmosphere<realT>::setAlphas( const realT & alpha)
{
   if(m_atm == nullptr)
   {
      std::cerr << "must set atmosphere pointer before setting alphas.\n";
      return -1;
   }
   
   return setAlphas( std::vector<realT>( m_atm->n_layers(), alpha));
}

template int boilingAtmosphere<float>::setAlphas(const realT & alpha);
template int boilingAtmosphere<double>::setAlphas(const realT & alpha);

template<typename realT>
int boilingAtmosphere<realT>::setAlphas( const std::vector<realT> & alphas)
{
   if(m_atm == nullptr)
   {
      std::cerr << "must set atmosphere pointer before setting alphas.\n";
      return -1;
   }
   
   if(alphas.size() != m_atm->n_layers())
   {
      std::cerr << "alpha vector must be same size as number of layers.\n";
      return -1;
   }
   
   m_alphas.resize(alphas.size());
   m_alphas1M.resize(alphas.size());
   
   for(int n=0; n<alphas.size();++n)
   {
      m_alphas[n] = alphas[n];
      m_alphas1M[n] = sqrt(1-pow(alphas[n],2));
   }
   
   return 0;
}

template int boilingAtmosphere<float>::setAlphas( const std::vector<float> & );
template int boilingAtmosphere<double>::setAlphas( const std::vector<double> & );

template<typename realT>
int boilingAtmosphere<realT>::genLayer( const size_t & n)
{
   for(size_t ii =0; ii < m_scrnSz; ++ii)
   {
      for(size_t jj=0; jj < m_scrnSz; ++jj)
      {
         m_noise[n](ii,jj) = complexT(normVar[n], normVar[n]);
      } 
   }

      
   fft_fwd(m_noise[n].data(), m_noise[n].data());
   
   //Apply the filter.
   m_noise[n] *= m_psdSqrt[n];
   
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
      
      m_transPhase[n] = m_noise[n];
   
   }
   
   return 0;
}

template int boilingAtmosphere<float>::genLayers();
template int boilingAtmosphere<double>::genLayers();

template<typename realT>
int boilingAtmosphere<realT>::updateLayers()
{
   
   //0) Zero out the phase screen
//   realT zero = 0;
   m_phase.setZero();
   
   static realT stepN = 1;

   improc::eigenImage<std::complex<realT>> windPhase;
   windPhase.resize(m_scrnSz, m_scrnSz);
      
   for(size_t n=0; n < m_atm->n_layers(); ++n)
   {
      
   
      #pragma omp parallel for
      for(size_t ii =0; ii < m_scrnSz; ++ii)
      {
         for(size_t jj=0; jj < m_scrnSz; ++jj)
         {
                        
            realT dk = mx::sigproc::freq_sampling(m_scrnSz, 0.5/( m_pupD/m_wfSz));
            realT kx = ii*dk;
            if(ii > 0.5*(m_scrnSz - 1)+1) kx = ((int)ii-(int) m_scrnSz)*dk;
            
            realT ky = jj*dk;
            if(jj > 0.5*(m_scrnSz - 1)+1) ky = ((int)jj-(int)m_scrnSz)*dk;
            
            realT vx = m_atm->layer_v_wind(n)*cos(m_atm->layer_dir(n));
            realT vy = m_atm->layer_v_wind(n)*sin(m_atm->layer_dir(n));
            
            std::complex<realT> phase = -two_pi<realT>()*stepN/m_fs*(kx*vx+ky*vy) * std::complex<realT>(0,1);
            windPhase(ii,jj) = exp(phase);
         }
      }
      m_windPhase[n] = windPhase;
   }
   stepN += 1.0;
   
#pragma omp parallel for
   for(size_t n=0; n<m_transPhase.size(); ++n)
   {
      mx::improc::eigenImage<std::complex<realT>> mtp;
      
      //1) Propagate by frozen flow.
      mtp = m_transPhase[n]*m_windPhase[n];
      
      //2) Multiply existing layer by alpha in the Fourier domain.
      //m_transPhase[n]*=m_alphas[n];
      
      //3) Generate new screen for this layer in the Fourier domain.
      //genLayer(n);
   
      //4) Multiply new layer by 1-alpha in the Fourier domain, and add it to the existing layer.
      //m_transPhase[n] += m_noise[n]*m_alphas1M[n];
      
      //5) Now Fourier transform to the space domain
      fft_inv(m_noise[n].data(), mtp.data());//m_transPhase[n].data());

      //6) And accumulate the real part as the new phase screen, weighted by Cn2.
      realT Cn2 = sqrt(m_atm->layer_Cn2(n));
      
#pragma omp critical
      m_phase += Cn2 * m_noise[n].real();
      
   }
   
   return 0;
}

template int boilingAtmosphere<float>::updateLayers();
template int boilingAtmosphere<double>::updateLayers();

template<typename realT>
int boilingAtmosphere<realT>::getWavefront( mx::improc::eigenImage<realT> & wf)
{
   //wf.resize( m_wfSz, m_wfSz);
   
   wf = m_phase.block(0,0, m_wfSz, m_wfSz);
   return 0;
}

template int boilingAtmosphere<float>::getWavefront(mx::improc::eigenImage<float> & wf);
template int boilingAtmosphere<double>::getWavefront(mx::improc::eigenImage<double> & wf);


