package com.guardhub.app

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.navigation.NavController
import androidx.navigation.fragment.NavHostFragment
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.navigateUp
import androidx.navigation.ui.onNavDestinationSelected
import androidx.navigation.ui.setupActionBarWithNavController
import com.guardhub.app.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var navController: NavController
    private lateinit var appBarConfiguration: AppBarConfiguration

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setSupportActionBar(binding.toolbar)

        val navHostFragment =
            supportFragmentManager.findFragmentById(R.id.nav_host_fragment) as NavHostFragment
        navController = navHostFragment.navController

        appBarConfiguration = AppBarConfiguration(
            setOf(R.id.homeFragment),
            binding.drawerLayout
        )

        setupActionBarWithNavController(navController, appBarConfiguration)

        // ✅ 사이드 메뉴 처리 (로그아웃 직접 처리)
        binding.navView.setNavigationItemSelectedListener { item ->

            when (item.itemId) {

                R.id.menu_logout -> {
                    // 🔥 실제 로그아웃 처리
                    getSharedPreferences("auth", MODE_PRIVATE)
                        .edit()
                        .clear()
                        .apply()

                    updateDrawerMenu()

                    navController.navigate(R.id.homeFragment)
                    binding.drawerLayout.closeDrawers()
                    true
                }

                else -> {
                    val handled = item.onNavDestinationSelected(navController)
                    if (handled) {
                        binding.drawerLayout.closeDrawers()
                    }
                    handled
                }
            }
        }

        updateDrawerMenu()
    }

    override fun onSupportNavigateUp(): Boolean {
        return navController.navigateUp(appBarConfiguration)
                || super.onSupportNavigateUp()
    }

    fun updateDrawerMenu() {
        val isLoggedIn = getSharedPreferences("auth", MODE_PRIVATE)
            .getBoolean("isLoggedIn", false)

        binding.navView.menu.clear()

        if (isLoggedIn) {
            binding.navView.inflateMenu(R.menu.drawer_menu_logged_in)
        } else {
            binding.navView.inflateMenu(R.menu.drawer_menu_logged_out)
        }
    }
}
