package com.guardhub.app

import android.content.Context
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.View
import androidx.fragment.app.Fragment
import com.guardhub.app.databinding.FragmentReportBinding

class fragment_report : Fragment(R.layout.fragment_report) {

    private var _binding: FragmentReportBinding? = null
    private val binding get() = _binding!!

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        _binding = FragmentReportBinding.bind(view)

        // 🔥 저장된 개인정보 자동 입력
        loadUserInfo()

        // 주민번호 앞자리 6자리 → 자동 이동
        binding.etReportResidentFront.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                if (s?.length == 6) {
                    binding.etReportResidentBack.requestFocus()
                }
            }
            override fun afterTextChanged(s: Editable?) {}
        })
    }

    private fun loadUserInfo() {
        val prefs = requireContext()
            .getSharedPreferences("user_prefs", Context.MODE_PRIVATE)
        val name =  prefs.getString("name", "")
        val phone =  prefs.getString("phone", "")
        val front = prefs.getString("resident_front", "")
        val back = prefs.getString("resident_back", "")

        binding.etReportName.setText(name)
        binding.etReportPhone.setText(phone)
        binding.etReportResidentFront.setText(front)
        binding.etReportResidentBack.setText(back)
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
