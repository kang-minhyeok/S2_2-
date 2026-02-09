package com.guardhub.app

import android.Manifest
import android.content.pm.PackageManager
import android.location.Location
import android.os.Bundle
import android.view.View
import android.widget.TextView
import androidx.core.app.ActivityCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationServices
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject

class fragment_local_culture : Fragment(R.layout.fragment_local_culture) {

    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private lateinit var locationText: TextView

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        locationText = view.findViewById(R.id.tvLocation)
        fusedLocationClient =
            LocationServices.getFusedLocationProviderClient(requireActivity())

        getUserLocation()
    }

    private fun getUserLocation() {
        if (
            ActivityCompat.checkSelfPermission(
                requireContext(),
                Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            locationText.text = "위치 권한이 필요합니다"
            return
        }

        fusedLocationClient.lastLocation
            .addOnSuccessListener { location: Location? ->
                if (location != null) {
                    convertLocationToAddress(location.latitude, location.longitude)
                } else {
                    locationText.text = "위치를 가져올 수 없습니다"
                }
            }
            .addOnFailureListener {
                locationText.text = "위치 조회 실패"
            }
    }

    private fun convertLocationToAddress(latitude: Double, longitude: Double) {
        viewLifecycleOwner.lifecycleScope.launch(Dispatchers.IO) {
            try {
                val client = OkHttpClient()

                val url =
                    "https://dapi.kakao.com/v2/local/geo/coord2address.json" +
                            "?x=$longitude&y=$latitude"

                val request = Request.Builder()
                    .url(url)
                    .addHeader(
                        "Authorization",
                        "KakaoAK ${BuildConfig.KAKAO_REST_API_KEY}"
                    )
                    .build()

                val response = client.newCall(request).execute()
                val body = response.body?.string()

                if (!response.isSuccessful || body.isNullOrEmpty()) {
                    throw Exception("API 응답 실패")
                }

                val json = JSONObject(body)
                val documents = json.optJSONArray("documents")

                if (documents == null || documents.length() == 0) {
                    throw Exception("주소 데이터 없음")
                }

                val addressObj =
                    documents.getJSONObject(0).optJSONObject("address")
                        ?: throw Exception("address 없음")

                val country = "대한민국"
                val region1 = addressObj.optString("region_1depth_name")
                val region2 = addressObj.optString("region_2depth_name")
                val region3 = addressObj.optString("region_3depth_name")

                val result = "$country - $region1 - $region2 - $region3"

                withContext(Dispatchers.Main) {
                    locationText.text = result
                }

            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    locationText.text = "주소 변환 실패"
                }
            }
        }
    }
}
