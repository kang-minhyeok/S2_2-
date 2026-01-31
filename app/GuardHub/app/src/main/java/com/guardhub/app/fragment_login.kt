package com.guardhub.app

import android.content.Context
import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController

class fragment_login : Fragment(R.layout.fragment_login) {

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val etId = view.findViewById<EditText>(R.id.etLoginId)
        val etPw = view.findViewById<EditText>(R.id.etLoginPassword)
        val btnLogin = view.findViewById<Button>(R.id.btnLogin)
        val tvSignup = view.findViewById<TextView>(R.id.tvGoSignup)

        // 🔥 회원 정보용
        val userPrefs = requireContext()
            .getSharedPreferences("user_prefs", Context.MODE_PRIVATE)

        // 🔥 로그인 상태용
        val authPrefs = requireContext()
            .getSharedPreferences("auth", Context.MODE_PRIVATE)

        btnLogin.setOnClickListener {

            val inputId = etId.text.toString().trim()
            val inputPw = etPw.text.toString()

            if (inputId.isEmpty() || inputPw.isEmpty()) {
                toast("아이디와 비밀번호를 입력하세요")
                return@setOnClickListener
            }

            val savedId = userPrefs.getString("userId", null)
            val savedPw = userPrefs.getString("password", null)

            // ===== 로그인 검증 =====
            if (inputId == savedId && inputPw == savedPw) {

                authPrefs.edit()
                    .putBoolean("isLoggedIn", true)
                    .apply()

                (requireActivity() as MainActivity).updateDrawerMenu()

                toast("로그인 성공")
                findNavController().navigate(R.id.homeFragment)

            } else {
                toast("아이디 또는 비밀번호가 올바르지 않습니다")
            }
        }

        tvSignup.setOnClickListener {
            findNavController().navigate(R.id.signupFragment)
        }
    }

    private fun toast(msg: String) {
        Toast.makeText(requireContext(), msg, Toast.LENGTH_SHORT).show()
    }
}
