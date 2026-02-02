package com.guardhub.app

import android.content.Context
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import com.guardhub.app.databinding.FragmentSignupBinding

class fragment_signup : Fragment() {

    private var _binding: FragmentSignupBinding? = null
    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentSignupBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // 주민번호 앞자리 6자리 → 자동 포커스 이동
        binding.etResidentFront.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}

            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                if (s?.length == 6) {
                    binding.etResidentBack.requestFocus()
                }
            }

            override fun afterTextChanged(s: Editable?) {}
        })

        // 회원가입 버튼
        binding.btnSignup.setOnClickListener {

            val id = binding.etId.text.toString().trim()
            val password = binding.etPassword.text.toString()
            val passwordCheck = binding.etPasswordCheck.text.toString()

            val residentFront = binding.etResidentFront.text.toString()
            val residentBack = binding.etResidentBack.text.toString()
            val name = binding.etSignupName.text.toString()
            val phone = binding.etSignupPhone.text.toString()

            if (id.isBlank() || password.isBlank() || passwordCheck.isBlank()) {
                toast("필수 항목을 입력하세요")
                return@setOnClickListener
            }

            if (password != passwordCheck) {
                toast("비밀번호가 일치하지 않습니다")
                return@setOnClickListener
            }

            if ((residentFront.length != 6 || residentBack.length != 7) && (residentFront.length != 0 || residentBack.length != 0)) {
                toast("주민번호를 정확히 입력하세요")
                return@setOnClickListener
            }

            // 🔥 실제 저장
            val prefs = requireContext()
                .getSharedPreferences("user_prefs", Context.MODE_PRIVATE)

            prefs.edit()
                .putString("userId", id)
                .putString("password", password)
                .putString("name", name)
                .putString("phone", phone)
                .putString("resident_front", residentFront)
                .putString("resident_back", residentBack)
                .apply()

            toast("회원가입 완료")

            findNavController()
                .navigate(R.id.action_signupFragment_to_loginFragment)
        }

    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
    private fun toast(msg: String) {
        Toast.makeText(requireContext(), msg, Toast.LENGTH_SHORT).show()
    }

}

