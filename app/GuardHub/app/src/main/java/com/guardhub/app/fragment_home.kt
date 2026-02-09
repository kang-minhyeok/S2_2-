package com.guardhub.app

import android.os.Bundle
import android.view.View
import android.widget.Button
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import com.guardhub.app.R

class fragment_home : Fragment(R.layout.fragment_home) {

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // 긴급 신고 버튼
        val btnEmergencyReport =
            view.findViewById<Button>(R.id.btnReport)

        btnEmergencyReport.setOnClickListener {
            // 신고 화면으로 이동
            findNavController().navigate(
                R.id.action_homeFragment_to_reportFragment
            )
        }
        val btn = view.findViewById<Button>(R.id.btnNews)

        btn.setOnClickListener {
            findNavController().navigate(
                R.id.action_homeFragment_to_neighborhoodNewsFragment
            )
        }
    }
}


