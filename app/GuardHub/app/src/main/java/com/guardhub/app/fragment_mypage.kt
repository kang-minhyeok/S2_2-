package com.guardhub.app

import android.content.Context
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import com.guardhub.app.databinding.FragmentMypageBinding

class fragment_mypage : Fragment() {

    private var _binding: FragmentMypageBinding? = null
    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentMypageBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val prefs = requireContext()
            .getSharedPreferences("user_prefs", Context.MODE_PRIVATE)

        val userid = prefs.getString("userId", "")
        val name =  prefs.getString("name", "")
        val phone =  prefs.getString("phone", "")
        val residentFront = prefs.getString("resident_front", "")
        val residentBack = prefs.getString("resident_back", "")

        binding.tvMyId.setText(userid)
        binding.etMyName.setText(name)
        binding.etMyPhone.setText(phone)
        binding.etResidentFront.setText(residentFront)
        binding.etResidentBack.setText(residentBack)

        // 🔥 앞자리 6자리 입력 → 자동 포커스 이동
        binding.etResidentFront.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}

            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                if (s?.length == 6) {
                    binding.etResidentBack.requestFocus()
                }
            }

            override fun afterTextChanged(s: Editable?) {}
        })

        // 🔥 수정 후 저장 버튼
        binding.btnSave.setOnClickListener {
            val name = binding.etMyName.text.toString()
            val phone = binding.etMyPhone.text.toString()
            val front = binding.etResidentFront.text.toString()
            val back = binding.etResidentBack.text.toString()

            if ((front.length != 6 || back.length != 7) && (front.length != 0 || back.length != 0)) {
                toast("주민번호를 정확히 입력해주세요")
                return@setOnClickListener
            }

            prefs.edit()
                .putString("name", name)
                .putString("phone", phone)
                .putString("resident_front", front)
                .putString("resident_back", back)
                .apply()

            toast("저장되었습니다")
        }
    }

    private fun toast(msg: String) {
        android.widget.Toast
            .makeText(requireContext(), msg, android.widget.Toast.LENGTH_SHORT)
            .show()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
